"""Expose backend-owned facts without asking the model to maintain a ledger."""
import asyncio
from collections import Counter
import hashlib
import json
import subprocess
import time
from typing import Annotated, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import Field


class AssistanceMiddleware(AgentMiddleware):
    def __init__(self, backend, model_limit, tool_limit, seconds):
        self.backend = backend
        self.model_limit, self.tool_limit, self.seconds = model_limit, tool_limit, seconds
        self.context = self.requests = None
        self.tool_names = []
        self.state, self.facts = {}, {}
        self.started = time.monotonic()
        self.tool_requests = 0
        self.goal = ''
        self.phase = 'ready'
        self.environment = None

    def before_agent(self, state, runtime):
        self.started, self.tool_requests, self.phase = time.monotonic(), 0, 'working'
        self.goal = next((m.text for m in reversed(state['messages']) if isinstance(m, HumanMessage)), '')
        self.backend.verification.begin_turn()
        try:
            self.environment = self.backend.environment_facts()
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.environment = {'scope': 'unknown', 'probe_error': type(error).__name__}
        with (self.backend.snapshot.session / 'environment-observations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(self.environment, ensure_ascii=False) + '\n')

    async def abefore_agent(self, state, runtime):
        await asyncio.to_thread(self.before_agent, state, runtime)

    def collect(self):
        """One current inventory also determines evidence freshness; no inferred dependencies."""
        ledger = self.backend.verification
        with self.backend._lock:
            files = ledger.files()
            hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
            baseline = self.backend.snapshot.manifest['files']
            version = ledger.fingerprint(files)
            changed = [{'path': name, 'change': 'added' if name not in baseline else
                        'deleted' if name not in hashes else 'modified'}
                       for name in sorted(hashes.keys() | baseline.keys())
                       if hashes.get(name) != baseline.get(name)]
            evidence = []
            for record in ledger.records():
                evidence.append({key: record.get(key) for key in
                    ('id', 'command', 'scope', 'kind', 'outcome', 'exit_code', 'timed_out', 'log_id')} |
                    {'current': record['before'] == record.get('after') == version,
                     'counts': record.get('test_evidence', {}).get('counts', {}),
                     'nodes_complete': record.get('test_evidence', {}).get('nodes_complete', False)})
            self.facts = {'version': version, 'changed_files': changed, 'evidence': evidence,
                          'accessible_file_count': len(hashes)}
        return self.facts

    def budgets(self, state):
        calls = max(state.get('run_model_call_count', 0), self.requests.calls if self.requests else 0)
        usage = [r.get('usage') for r in (self.requests.records if self.requests else [])]
        policy = self.context.policy if self.context else None
        tokens = state.get('context_estimate') if self.context else None
        return {'model_calls_remaining_including_next': max(0, self.model_limit - calls),
                'tool_requests_remaining_estimate': max(0, self.tool_limit - self.tool_requests),
                'seconds_remaining': max(0, int(self.seconds - (time.monotonic() - self.started))),
                'context_input_estimate': tokens,
                'context_input_limit': policy.input_limit if policy else None,
                'context_headroom_estimate': max(0, policy.input_limit - tokens)
                    if policy and tokens is not None else None,
                'reported_main_request_tokens': sum(u.get('total_tokens', 0) for u in usage if u),
                'main_requests_without_usage': sum(not u for u in usage),
                'total_token_budget_remaining': None}

    def summary(self, state):
        evidence = self.facts.get('evidence', [])
        changed = self.facts.get('changed_files', [])
        counts = Counter(('current_' if e['current'] else 'historical_') + e['outcome'] for e in evidence)
        return {'phase': self.phase, 'task_success': None, 'budget': self.budgets(state),
                'code_version': self.facts.get('version'), 'changed_file_count': len(changed),
                'changed_files': changed[:8], 'evidence_counts': dict(counts),
                'recent_evidence': [{k: e[k] for k in ('id', 'current', 'outcome', 'log_id')}
                                    for e in evidence[-4:]],
                'details_tool': 'get_task_state'}

    def suffix(self, state):
        environment = '' if self.environment is None else (
            '\nTurn-start environment (read-only observations, not permission; null means unknown):\n'
            + json.dumps(self.environment, ensure_ascii=False, separators=(',', ':'))
            + '\nAgent API tools are not shell executables. Use search_files/edit_file/show_changes for search/edit/diff; '
              'their availability does not imply shell rg/apply_patch/git metadata. '
              'Avoid commands reported unavailable. Facts may change after commands; inspect new errors.\n')
        return environment + '\nProgram observations (facts, not semantic acceptance):\n' + json.dumps(
            self.summary(state), ensure_ascii=False, separators=(',', ':'))

    async def abefore_model(self, state, runtime):
        self.state = state
        await asyncio.to_thread(self.collect)

    async def awrap_model_call(self, request, handler):
        self.state = request.state
        system = request.system_message or SystemMessage(content='')
        updated = SystemMessage(content=[*system.content_blocks, {'type': 'text', 'text': self.suffix(request.state)}])
        record = self.summary(request.state)
        with (self.backend.snapshot.session / 'assistance-state.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        # ContextMiddleware is inside this wrapper, so its final size guard includes the facts.
        return await handler(request.override(system_message=updated))

    async def awrap_tool_call(self, request, handler):
        self.tool_requests += 1
        return await handler(request)

    async def aafter_agent(self, state, runtime):
        self.phase, self.state = 'reply_returned', state
        await asyncio.to_thread(self.collect)
        (self.backend.snapshot.session / 'assistance-final.json').write_text(
            json.dumps(self.summary(state), ensure_ascii=False, indent=2), encoding='utf-8')

    def status_tool(self):
        @tool
        def get_task_state(section: Literal['summary', 'files', 'evidence', 'access'] = 'summary',
                           offset: Annotated[int, Field(ge=0)] = 0,
                           limit: Annotated[int, Field(ge=1, le=40)] = 20) -> str:
            """Read current program facts, never run tests. Files are changes since session start; evidence shows exact-code freshness and original outcomes. Paginate files/evidence. Access describes tool and permission boundaries. Estimates and unknown values are labeled; task_success is not inferred."""
            self.collect()
            if section in {'files', 'evidence'}:
                values = self.facts['changed_files' if section == 'files' else 'evidence']
                result = {'items': values[offset:offset + limit], 'total': len(values),
                          'next_offset': offset + limit if offset + limit < len(values) else None}
            elif section == 'access':
                result = {'workspace_mode': self.backend.snapshot.mode,
                          'path_scope': 'Workspace-relative paths, subject to backend protected/ignored-file checks; list/search/read for actual accessible contents.',
                          'accessible_file_count': self.facts['accessible_file_count'],
                          'tools': self.tool_names,
                          'turn_start_environment': self.environment,
                          'permission_authority': 'Existing backend approval policy; this status never grants approval.',
                          'execution_rules': getattr(self.backend, 'execution_rules', None),
                          'recovery_write_review': getattr(self.backend, 'review_writes', False),
                          'container_direct_mode': 'Read commands use a read-only workspace mount; writable commands and deletion of preexisting files need approval.'
                              if self.backend.snapshot.mode == 'direct' and not getattr(self.backend, 'execution_rules', None) else None,
                          'isolated_mode': 'Edits and commands affect only the isolated copy; no network or automatic export to the original.'
                              if self.backend.snapshot.mode == 'isolated' else None}
            else:
                result = self.summary(self.state) | {'current_user_request_excerpt': self.goal[:1200],
                    'request_excerpt_truncated': len(self.goal) > 1200,
                    'completion_evidence': 'Use the original user request as the acceptance contract. Report actual changes, current relevant checks, known failures and unverified requirements. Passing receipts prove only their checks; requirement coverage remains a model judgment.'}
            return json.dumps(result, ensure_ascii=False)
        return get_task_state


def assisted_tools(backend, tools, assistance):
    @tool
    def run_verification(command: str, scope: str = '', timeout_seconds: int = 60,
                         kind: Literal['test', 'check'] = 'test') -> str:
        """Run one check and automatically save its code version, exit status, test counts and log. No registration or evidence binding needed. kind=check supports assertion/build commands; zero exit alone is not test success. scope optionally describes what is checked, not proven coverage. Existing backend permissions apply; use read_tool_log for full output."""
        try:
            return backend.run_verification(command, scope or command, timeout_seconds, kind, [])
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            return f"工具错误：{type(error).__name__}: {error if isinstance(error, ValueError) else '执行未成功，不会自动重试'}"
    removed = {'register_behavior', 'bind_verification', 'replace_verification', 'review_work', 'run_verification'}
    result = [t for t in tools if t.name not in removed] + [run_verification, assistance.status_tool()]
    assistance.tool_names = [t.name for t in result]
    return result
