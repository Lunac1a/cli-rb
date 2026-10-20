"""Program-owned execution evidence; never equate passing tests with task correctness."""

import asyncio
from copy import deepcopy
from difflib import get_close_matches
import hashlib
import json
import re
from typing import NotRequired
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from cli_rb.workspace.patches import inventory, file_bytes
from cli_rb.workspace.container_worker import python_edit_warnings, python_syntax_errors
from cli_rb.quality.test_evidence import parse_test_evidence, resolve_evidence, evidence_summary
from cli_rb.workspace.command_feedback import command_feedback
from cli_rb.quality.evidence_lifecycle import startup_failure, resolved_startups


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def parse_test_outcome(result):
    """Recognize test-runner summaries, not arbitrary zero-exit commands."""
    if result.get('timed_out') or result.get('exit_code') != 0:
        return 'failed'
    output = re.sub(r'\x1b\[[0-9;]*m', '', result.get('output', ''))
    if re.search(r'\b[1-9]\d* (?:failed|errors?)\b|^FAILED\b|^ERROR\b', output, re.M):
        return 'failed'
    evidence = parse_test_evidence(result)
    counts = evidence['counts']
    if counts:
        if counts['failed'] or counts['errors']:
            return 'failed'
        if evidence['framework'] == 'sympy' and counts['xpassed']:
            return 'failed'
        if counts['passed']:
            return 'passed'
    return 'unknown'


class VerificationLedger:
    def __init__(self, backend):
        self.backend = backend
        self.path = backend.snapshot.session / 'verification.jsonl'
        self.baseline = digest(backend.snapshot.manifest['files'])
        self.turn_start = self.baseline
        self.turn_record_count = 0
        self.plan_path = backend.snapshot.session / 'verification-plan.jsonl'
        self.turn_plan_count = 0
        self.binding_path = backend.snapshot.session / 'behavior-bindings.jsonl'
        self.turn_binding_count = 0
        self.reconciliation_path = backend.snapshot.session / 'evidence-replacements.jsonl'

    def files(self):
        snapshot = self.backend.snapshot
        ignored = set(snapshot.manifest['excluded'])
        from cli_rb.workspace.ignore_rules import snapshot_ignored_paths
        ignored |= snapshot_ignored_paths(snapshot)
        files = inventory(snapshot.workspace, ignored,
                          max_file_bytes=snapshot.manifest.get('max_file_bytes', 1024 * 1024),
                          max_total_bytes=snapshot.manifest.get('max_total_bytes', 64 * 1024 * 1024))
        return files

    def fingerprint(self, files=None):
        return digest({name: hashlib.sha256(data).hexdigest()
                       for name, data in (self.files() if files is None else files).items()})

    def begin_turn(self):
        with self.backend._lock:
            self.turn_start = self.fingerprint()
            self.turn_record_count = len(self.records())
            self.turn_plan_count = len(self.plans())
            self.turn_binding_count = len(self.bindings())

    def plans(self):
        return [json.loads(line) for line in self.plan_path.read_text(encoding='utf-8').splitlines()] if self.plan_path.exists() else []

    def record_plan(self, todos):
        with self.backend._lock:
            todos = deepcopy(todos)
            existing = self.requirements()
            seen = set()
            for todo in todos:
                requirement = todo.get('verification')
                revise = todo.pop('revise_verification', False)
                if requirement:
                    key = requirement.get('id')
                    if key in seen:
                        raise ValueError('Each verification id must occur only once in a plan.')
                    seen.add(key)
                    if key in existing and not revise:
                        if any(value != existing[key].get(field) for field, value in requirement.items()):
                            raise ValueError('Existing behavior definition changed. For status/content updates send verification with id only; use revise_verification=true only for a real requirement change, which invalidates its evidence.')
                        requirement = todo['verification'] = deepcopy(existing[key])
                if requirement and not all(requirement.get(key) for key in ('id', 'expected', 'assertion')):
                    raise ValueError('Verification entries require nonempty id, expected behavior, and assertion; affected_paths may be empty pending investigation.')
                if requirement and 'affected_paths' not in requirement:
                    raise ValueError('New or revised behaviors require affected_paths, which may be empty pending investigation.')
            with self.plan_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(todos, ensure_ascii=False) + '\n')
            return todos

    def requirements(self):
        # Removing a todo does not silently erase its critical behavior. Updating
        # its contract is allowed, but previous evidence no longer matches it.
        requirements = {}
        for todos in self.plans()[self.turn_plan_count:]:
            for todo in todos:
                if todo.get('verification'):
                    requirement = todo['verification']
                    requirements[requirement['id']] = requirement
        return requirements

    def register_behavior(self, behavior_id, expected, assertion, affected_paths):
        """Upsert one product contract independently of progress; keep historical plans readable."""
        definition = {'id': behavior_id, 'expected': expected, 'assertion': assertion,
                      'affected_paths': affected_paths}
        self.record_plan([{'verification': definition, 'revise_verification': True}])
        return json.dumps({'behavior': definition, 'contract': digest(definition),
                           'note': 'Progress updates do not alter this definition. Bind existing current evidence; rerun only for a real coverage gap.'})

    def bindings(self):
        return [json.loads(line) for line in self.binding_path.read_text(encoding='utf-8').splitlines()] if self.binding_path.exists() else []

    def bind_verification(self, behavior_id, evidence_id, test_nodes, assertion):
        with self.backend._lock:
            requirement = self.requirements().get(behavior_id)
            record = next((r for r in self.records()[self.turn_record_count:] if r['id'] == evidence_id), None)
            if not requirement or not record or not assertion.strip():
                raise ValueError('Use a registered behavior, an evidence ID from this turn, and the actual assertion explanation.')
            if record.get('purpose') == 'reproduce':
                raise ValueError('A reproduction is not post-fix verification.')
            current = self.fingerprint()
            if record['before'] != current or record['after'] != current:
                raise ValueError('Evidence is stale or code changed during execution; cannot bind it to current code.')
            details = record.get('test_evidence', {})
            if record['kind'] == 'test':
                if not test_nodes or not details.get('nodes_complete') or any(n not in details['nodes'] for n in test_nodes):
                    raise ValueError('Select exact nodes from a complete test receipt; empty, guessed or unobserved nodes cannot establish coverage.')
            elif test_nodes:
                raise ValueError('Command checks have no test nodes; use an empty list.')
            binding = {'behavior_id': behavior_id, 'contract': digest(requirement),
                       'evidence_id': evidence_id, 'test_nodes': sorted(set(test_nodes)), 'assertion': assertion}
            with self.binding_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(binding, ensure_ascii=False) + '\n')
            return json.dumps({**binding, 'note': 'Binding saved without rerunning. Node results and all other failures remain in review_work.'})

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()] if self.path.exists() else []

    def replace_verification(self, old_id, new_id, old_nodes, new_nodes, reason):
        """Record an explicit coverage claim for renamed/replaced tests, never erase a run."""
        with self.backend._lock:
            records = {r['id']: r for r in self.records()[self.turn_record_count:]}
            old, new = records.get(old_id), records.get(new_id)
            if not old or not new or old_id == new_id or not reason.strip():
                raise ValueError('Use distinct evidence IDs from this turn and explain the replacement assertions.')
            if old.get('purpose') == 'reproduce' or new.get('purpose') == 'reproduce':
                raise ValueError('Reproductions must be linked through their registered behavior contract.')
            old_details, new_details = old.get('test_evidence', {}), new.get('test_evidence', {})
            check_migration = old['kind'] == 'check' and not old_nodes
            if check_migration:
                contracts = old.get('behavior_contracts', {})
                if (not contracts or not old.get('assertion') or old.get('timed_out')
                        or old.get('before') != old.get('after') or old.get('exit_code') is None
                        or any(new.get('behavior_contracts', {}).get(k) != v for k, v in contracts.items())):
                    raise ValueError('Check migration requires a stable completed check with declared assertions and matching behavior contracts in the new test run.')
            if (not new_nodes or not new_details.get('nodes_complete') or new['kind'] != 'test'
                    or not check_migration and (not old_nodes or not old_details.get('nodes_complete') or old['kind'] != 'test')):
                raise ValueError('Select exact old nodes and passing replacement nodes from complete test receipts.')
            invalid = []
            for side, selected, details in [('old', old_nodes, old_details), ('new', new_nodes, new_details)]:
                available = [n for n, status in details.get('nodes', {}).items()
                             if side == 'old' or status == 'passed']
                for node in selected:
                    if node not in available:
                        invalid.append({'side': side, 'node': node,
                                        'observed_status': details['nodes'].get(node, 'not_found'),
                                        'candidates': get_close_matches(node, available, n=3) or available[:3]})
            if invalid:
                raise ValueError('Invalid test nodes: ' + json.dumps(invalid[:5], ensure_ascii=False) +
                                 '. Copy exact IDs from the receipt; candidates are suggestions, not automatic replacements.')
            current = self.fingerprint()
            if new['before'] != current or new['after'] != current or new['outcome'] != 'passed':
                raise ValueError('Replacement must be a successful test run on the current code.')
            if list(records).index(new_id) <= list(records).index(old_id):
                raise ValueError('Replacement evidence must be newer than the old run.')
            link = {'old_id': old_id, 'new_id': new_id, 'old_nodes': sorted(set(old_nodes)),
                    'new_nodes': sorted(set(new_nodes)), 'reason': reason, 'current': current,
                    'check_migration': check_migration}
            with self.reconciliation_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(link, ensure_ascii=False) + '\n')
            return json.dumps({**link, 'note': 'Coverage is your explicit claim; original results and unrelated failures remain.'})

    def reconciliations(self, current, records):
        if not self.reconciliation_path.exists():
            return []
        ids = {r['id'] for r in records}
        return [link for line in self.reconciliation_path.read_text(encoding='utf-8').splitlines()
                if (link := json.loads(line))['current'] == current
                and link['old_id'] in ids and link['new_id'] in ids]

    def run(self, command, scope, timeout_seconds, kind='test', behavior_ids=None,
            assertion='', purpose='verify', expected_failure=''):
        if not scope.strip():
            raise ValueError('scope must describe the behavior checked; it is a model claim, not proof of coverage.')
        if kind not in {'test', 'check'}:
            raise ValueError('kind must be test or check.')
        if purpose not in {'verify', 'regression', 'reproduce'}:
            raise ValueError('purpose must be verify, regression, or reproduce.')
        if purpose == 'reproduce' and not expected_failure.strip():
            raise ValueError('Pre-fix reproduction requires expected_failure error text declared before execution.')
        with self.backend._lock:
            requirements = self.requirements()
            behavior_ids = behavior_ids or []
            if any(key not in requirements for key in behavior_ids) or (behavior_ids and not assertion.strip()):
                raise ValueError('First register product behaviors with register_behavior and describe the assertion used by this actual check.')
            before = self.fingerprint()
            record = {'id': 'verification-' + uuid4().hex, 'command': command,
                      'scope': scope, 'kind': kind, 'before': before, 'after': None, 'outcome': 'unknown',
                      'purpose': purpose, 'assertion': assertion, 'expected_failure': expected_failure,
                      'behavior_contracts': {key: digest(requirements[key]) for key in behavior_ids}}
            count = len(self.backend.command_evidence)
            feedback = None
            try:
                response = self.backend.verification_command(command, timeout_seconds, kind == 'test')
                # Read the full host-owned log, even when the model-facing output is truncated.
                if len(self.backend.command_evidence) > count:
                    entry = self.backend.command_evidence[-1]
                    record['log_id'] = entry['log_id']
                    raw = (self.backend.logs / (entry['log_id'] + '.txt')).read_text(encoding='utf-8')
                    try:
                        result = json.loads(raw)
                        feedback = json.loads(command_feedback(raw, entry['log_id']))
                        details = parse_test_evidence(result)
                        if kind == 'test':
                            record['startup_failure'] = startup_failure(command, result)
                        outcome = parse_test_outcome(result)
                        if details.get('source') == 'junit':
                            outcome = ('passed' if result.get('exit_code') == 0 and not result.get('timed_out')
                                       and details['nodes_complete'] and details['counts']['passed']
                                       and not details['counts']['failed'] and not details['counts']['errors']
                                       else 'failed' if result.get('exit_code') != 0 or result.get('timed_out')
                                       else 'unknown')
                        if kind == 'check' and outcome == 'unknown' and result.get('exit_code') == 0 and not result.get('timed_out'):
                            outcome = 'command_succeeded'
                        record.update(outcome=outcome, exit_code=result.get('exit_code'),
                                      timed_out=result.get('timed_out'), test_evidence=details,
                                      executed_command=result.get('executed_command', command),
                                      expected_failure_observed=bool(purpose == 'reproduce' and
                                          result.get('exit_code') not in (None, 0) and not result.get('timed_out')
                                          and expected_failure in result.get('output', '')))
                    except (ValueError, TypeError, AttributeError):
                        record['outcome'] = 'unknown'
                else:
                    # A denied attempt has an audit receipt, but no execution evidence.
                    # Docker may instead return a plain-text permission refusal.
                    try:
                        refusal = json.loads(response)
                    except (ValueError, TypeError):
                        refusal = None
                    if isinstance(refusal, dict) and refusal.get('denied') and refusal.get('started') is False:
                        feedback = refusal
                        record.update(log_id=refusal['log_id'], exit_code=None,
                                      timed_out=False, denied=True, started=False)
                record['after'] = self.fingerprint()
            finally:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            _, replacements = resolve_evidence(self.records()[self.turn_record_count:])
            return json.dumps({**record, 'command_result': feedback, 'replacements': [item for item in replacements
                               if item['new_id'] == record['id']]}, ensure_ascii=False)

    def assess(self, require_behaviors=False, *, files=None):
        with self.backend._lock:
            current = self.fingerprint(files)
            records = self.records()
            requirements = self.requirements()
            turn_records = records[self.turn_record_count:]
            active_records, startup_links = resolved_startups(turn_records, current)
            latest, replacements = resolve_evidence(active_records, self.reconciliations(current, active_records))
            replacements.extend(startup_links)
            issues = []
            if current == self.turn_start and len(records) == self.turn_record_count and not requirements:
                return {'status': 'not_applicable', 'current': current, 'issues': [], 'evidence_ids': []}
            if current != self.turn_start and not latest:
                issues.append('Current changes have no evidence saved by run_verification.')
            elif current != self.turn_start and not any(
                    item['record']['before'] == item['record']['after'] == current
                    and item['record']['outcome'] in {'passed', 'command_succeeded'} for item in latest):
                issues.append('Current changes have no current successful verification.')
            behaviors = []
            for key, requirement in requirements.items():
                matches = [r for r in active_records
                           if r.get('purpose') != 'reproduce' and
                           r.get('behavior_contracts', {}).get(key) == digest(requirement)]
                record = matches[-1] if matches else None
                ok = bool(requirement['affected_paths'] and record and record['before'] == record['after'] == current and
                          record['outcome'] in {'passed', 'command_succeeded'})
                if ok and record['kind'] == 'test':
                    details = record.get('test_evidence', {})
                    ok = bool(details.get('nodes_complete') and
                              all(status == 'passed' for status in details['nodes'].values()))
                bindings = [b for b in self.bindings()[self.turn_binding_count:]
                            if b['behavior_id'] == key and b['contract'] == digest(requirement)]
                # A correction for the same receipt replaces its selection; separate
                # receipts contribute together instead of erasing earlier coverage.
                selected = {b['evidence_id']: b for b in bindings}
                binding = bindings[-1] if bindings else None
                observed = {}
                if selected:
                    ok = bool(requirement['affected_paths'])
                    for evidence_id, selection in selected.items():
                        source = next(r for r in records if r['id'] == evidence_id)
                        if source['kind'] == 'test':
                            for candidate in turn_records:
                                details = candidate.get('test_evidence', {})
                                for node in selection['test_nodes']:
                                    if node in details.get('nodes', {}):
                                        observed[node] = (candidate, details['nodes'][node])
                            ok = ok and all(node in observed and observed[node][1] == 'passed'
                                            and observed[node][0]['before'] == observed[node][0]['after'] == current
                                            and observed[node][0].get('test_evidence', {}).get('nodes_complete')
                                            and not observed[node][0].get('timed_out')
                                            for node in selection['test_nodes'])
                        else:
                            current_check = next((r for r in reversed(turn_records)
                                                  if r['kind'] == 'check' and r['command'] == source['command']), source)
                            ok = ok and current_check['before'] == current_check['after'] == current and current_check['outcome'] in {'passed', 'command_succeeded'}
                    record = next(r for r in records if r['id'] == binding['evidence_id'])
                behaviors.append({**requirement, 'status': 'evidence_current' if ok else 'unverified',
                                  'evidence_id': record['id'] if record else None,
                                  'evidence_ids': list(selected) if selected else [record['id']] if record else [],
                                  'assertion_claim': binding['assertion'] if binding else record.get('assertion') if record else None,
                                  'assertion_claims': [b['assertion'] for b in selected.values()],
                                  'bound_nodes': sorted({n for b in selected.values() for n in b['test_nodes']}) if selected else None,
                                  'node_evidence': {n: r['id'] for n, (r, _) in observed.items()}})
                if not ok:
                    reason = 'Affected paths still need investigation' if not requirement['affected_paths'] else 'No current complete passing evidence'
                    issues.append(f'Key behavior {key}: {reason}: {requirement["expected"]}')
            if require_behaviors and current != self.turn_start and not requirements:
                issues.append('Code changed, but no product behaviors and assertions were registered; their verification cannot be confirmed.')
            evidence = []
            for item in latest:
                record, nodes = item['record'], item['nodes']
                is_current = record['before'] == current and record['after'] == current
                outcome = record['outcome']
                if nodes is not None:
                    outcome = ('failed' if any(s in {'failed', 'errors'} for s in nodes.values()) else
                               'passed' if 'passed' in nodes.values() else
                               'expected_failure' if set(nodes.values()) == {'xfailed'} else 'unknown')
                if record.get('purpose') == 'reproduce':
                    issues.append(f"{record['id']}: pre-fix reproduction is not yet linked to current passing evidence.")
                evidence.append({**record, 'current': is_current, 'remaining_nodes': nodes,
                                 'remaining_outcome': outcome})
                historical_success = (not is_current and record['kind'] == 'check'
                                      and outcome in {'passed', 'command_succeeded'}
                                      and record['before'] == record['after'] and not record.get('timed_out'))
                if not is_current and not historical_success:
                    issues.append(f"{record['id']}: verification does not match current code, or code changed during execution.")
                elif outcome not in {'passed', 'command_succeeded', 'expected_failure'}:
                    issues.append(f"{record['id']}: remaining outcome is {outcome}; it is not passing evidence.")
            return {'status': 'needs_attention' if issues else ('evidence_current' if latest else 'not_applicable'),
                    'current': current, 'issues': issues,
                    'evidence_ids': [item['record']['id'] for item in latest],
                    'evidence': evidence, 'replacements': replacements,
                    'behaviors': behaviors,
                    'completion_status': ('partial_unverified' if issues else
                                          'declared_behaviors_evidenced' if behaviors else 'checks_only'),
                    'coverage': 'scope is declared by the model; it does not prove complete requirement coverage or code correctness'}

    def review_changes(self, *, files=None):
        """Read-only review shared by explicit tools and the final stop hook."""
        snapshot = self.backend.snapshot
        current = self.files() if files is None else files
        original = snapshot.manifest['files']
        changes, risks, errors = [], [], []
        for name in sorted(original.keys() | current.keys()):
            data = current.get(name)
            if data is not None and hashlib.sha256(data).hexdigest() == original.get(name):
                continue
            changes.append(name)
            if name.endswith('.py'):
                before = file_bytes(self.backend.patches.baseline, name, self.backend.patches.max_file_bytes) if name in original else None
                syntax = python_syntax_errors(name, data)
                errors.extend(syntax)
                if not syntax:
                    risks.extend(python_edit_warnings(name, before, data))
        return {'changed_files': changes, 'risks': risks, 'syntax_errors': errors}

    def review(self):
        with self.backend._lock:
            files = self.files()
            report = self.assess(require_behaviors=True, files=files)
            if report['status'] != 'not_applicable':
                report.update(self.review_changes(files=files))
                report['issues'].extend(report['syntax_errors'])
                if report['issues']:
                    report.update(status='needs_attention', completion_status='partial_unverified')
            return report


class CompletionState(AgentState):
    completion_feedback_count: NotRequired[int]


class CompletionMiddleware(AgentMiddleware):
    state_schema = CompletionState

    def __init__(self, backend, model_call_limit):
        self.backend = backend
        self.model_call_limit = model_call_limit

    def before_agent(self, state, runtime):
        self.backend.verification.begin_turn()
        return {'completion_feedback_count': 0}

    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)

    def review(self, state):
        from cli_rb.runtime.agent import has_reply
        last = state['messages'][-1]
        if not isinstance(last, AIMessage) or not has_reply(last):
            return None
        try:
            report = self.backend.verification.review()
        except (OSError, ValueError, RuntimeError) as exc:
            report = {'status': 'needs_attention', 'issues': [f'Evidence assessment failed: {type(exc).__name__}'],
                      'evidence_ids': []}
        feedback = (report['issues'] and not state.get('completion_feedback_count', 0)
                    and state.get('run_model_call_count', 0) < self.model_call_limit)
        record = {**report, 'feedback_sent': bool(feedback),
                  'feedback_decision': ('sent' if feedback else
                                        'no_issues' if not report['issues'] else
                                        'feedback_already_used' if state.get('completion_feedback_count', 0) else
                                        'model_call_limit'),
                  'model_calls_at_check': state.get('run_model_call_count'),
                  'model_call_limit': self.model_call_limit}
        with (self.backend.snapshot.session / 'completion-checks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if feedback:
            return {'messages': [HumanMessage(content='Program completion check (at most one feedback turn):\n' +
                     '\n'.join(report['issues']) +
                     '\nProgram verification summary:\n' + evidence_summary(report) +
                     '\nWhen necessary and within budget, add verification with run_verification; otherwise explicitly report incomplete or unverified work. '
                     'Do not delete tests or narrow requirements merely to clear this message. Unrecognized test output remains unknown. '
                     'Do not rerun unchanged checks for plan wording updates. A suspected environment or pre-existing failure remains unclassified unless a comparable pre-patch run demonstrates it; report that uncertainty. '
                     'Your earlier draft may not have been displayed. Return a complete standalone final answer, including requested usage commands and deliverables, not a correction referring to a previous message. Explain concrete test failures or limitations in plain language without internal evidence IDs or bookkeeping terminology.')],
                    'completion_feedback_count': 1, 'jump_to': 'model'}
        reply = last.model_copy(deep=True)
        reply.response_metadata['completion_check'] = report
        if report['status'] != 'not_applicable':
            label = ('验证证据未通过核对：' + '；'.join(report['issues']) if report['issues'] else
                     '已记录验证与当前代码一致；不代表需求覆盖完整。')
            # Keep provider reasoning/tool blocks and usage intact; append only visible text.
            state_label = ('部分完成／关键验证未完成' if report.get('completion_status') == 'partial_unverified' else
                           '已登记关键行为有当前证据（不等同任务语义验收）' if report.get('behaviors') else
                           '已记录检查，未建立完整行为覆盖')
            suffix = '\n\n[程序核对] ' + state_label + '。' + label + '\n' + evidence_summary(report)
            for behavior in report.get('behaviors', []):
                suffix += f"\n- 行为 {behavior['id']}：{behavior['expected']}；{behavior['status']}；证据 {behavior['evidence_id']}"
            if isinstance(reply.content, str):
                reply.content = '[完成状态] ' + state_label + '\n\n' + reply.content + suffix
            else:
                reply.content.insert(0, {'type': 'text', 'text': '[完成状态] ' + state_label + '\n\n'})
                reply.content.append({'type': 'text', 'text': suffix})
        return {'messages': [reply]}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state, runtime):
        return self.review(state)

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state, runtime):
        return await asyncio.to_thread(self.review, state)
