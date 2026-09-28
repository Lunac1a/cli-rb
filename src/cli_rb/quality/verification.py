"""Program-owned execution evidence; never equate passing tests with task correctness."""

import asyncio
import hashlib
import json
import re
from typing import NotRequired
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from cli_rb.workspace.patches import inventory, file_bytes
from cli_rb.workspace.container_worker import python_edit_warnings


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def parse_test_outcome(result):
    """Recognize pytest/unittest summaries, not arbitrary zero-exit commands."""
    if result.get('timed_out') or result.get('exit_code') != 0:
        return 'failed'
    output = re.sub(r'\x1b\[[0-9;]*m', '', result.get('output', ''))
    if re.search(r'\b[1-9]\d* (?:failed|errors?)\b|^FAILED\b|^ERROR\b', output, re.M):
        return 'failed'
    if re.search(r'\b[1-9]\d* passed\b', output):
        return 'passed'
    count = re.search(r'^Ran (\d+) tests? in ', output, re.M)
    if count and int(count[1]) > 0 and re.search(r'^OK(?:\s*\([^\n]*\))?\s*$', output, re.M):
        skipped = re.search(r'^OK \(skipped=(\d+)\)', output, re.M)
        if not skipped or int(skipped[1]) < int(count[1]):
            return 'passed'
    return 'unknown'


class VerificationLedger:
    def __init__(self, backend):
        self.backend = backend
        self.path = backend.snapshot.session / 'verification.jsonl'
        self.baseline = digest(backend.snapshot.manifest['files'])
        self.turn_start = self.baseline
        self.turn_record_count = 0

    def files(self):
        snapshot = self.backend.snapshot
        ignored = set(snapshot.manifest['excluded'])
        if snapshot.mode == 'direct':
            from cli_rb.workspace.live_workspace import ignored_paths
            ignored |= ignored_paths(snapshot.workspace) - snapshot.manifest['files'].keys()
        files = inventory(snapshot.workspace, ignored,
                          max_file_bytes=snapshot.manifest.get('max_file_bytes', 1024 * 1024))
        return files

    def fingerprint(self):
        return digest({name: hashlib.sha256(data).hexdigest() for name, data in self.files().items()})

    def begin_turn(self):
        with self.backend._lock:
            self.turn_start = self.fingerprint()
            self.turn_record_count = len(self.records())

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()] if self.path.exists() else []

    def run(self, command, scope, timeout_seconds, kind='test'):
        if not scope.strip():
            raise ValueError('scope 必须说明本次检查的行为范围；它是模型声明，不是覆盖率证明。')
        if kind not in {'test', 'check'}:
            raise ValueError('kind 必须是 test 或 check。')
        with self.backend._lock:
            before = self.fingerprint()
            record = {'id': 'verification-' + uuid4().hex, 'command': command,
                      'scope': scope, 'kind': kind, 'before': before, 'after': None, 'outcome': 'unknown'}
            count = len(self.backend.command_evidence)
            try:
                self.backend.run_command(command, timeout_seconds)
                # Read the full host-owned log, even when the model-facing output is truncated.
                if len(self.backend.command_evidence) > count:
                    entry = self.backend.command_evidence[-1]
                    record['log_id'] = entry['log_id']
                    raw = (self.backend.logs / (entry['log_id'] + '.txt')).read_text(encoding='utf-8')
                    try:
                        result = json.loads(raw)
                        outcome = parse_test_outcome(result)
                        if kind == 'check' and outcome == 'unknown' and result.get('exit_code') == 0 and not result.get('timed_out'):
                            outcome = 'command_succeeded'
                        record.update(outcome=outcome, exit_code=result.get('exit_code'),
                                      timed_out=result.get('timed_out'))
                    except (ValueError, TypeError, AttributeError):
                        record['outcome'] = 'unknown'
                record['after'] = self.fingerprint()
            finally:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            return json.dumps(record, ensure_ascii=False)

    def assess(self):
        with self.backend._lock:
            current = self.fingerprint()
            records = self.records()
            latest = {r['scope']: r for r in records[self.turn_record_count:]}
            issues = []
            if current == self.turn_start and len(records) == self.turn_record_count:
                return {'status': 'not_applicable', 'current': current, 'issues': [], 'evidence_ids': []}
            if current != self.turn_start and not latest:
                issues.append('当前修改没有 run_verification 保存的验证证据。')
            for scope, record in latest.items():
                if record['before'] != current or record['after'] != current:
                    issues.append(f"{record['id']}：验证与当前代码不一致（或执行期间代码变化）。")
                elif record['outcome'] not in {'passed', 'command_succeeded'}:
                    issues.append(f"{record['id']}：验证结果为 {record['outcome']}，不能作为通过证据。")
            return {'status': 'needs_attention' if issues else ('evidence_current' if latest else 'not_applicable'),
                    'current': current, 'issues': issues,
                    'evidence_ids': [r['id'] for r in latest.values()],
                    'coverage': 'scope由模型声明；不证明需求完整覆盖或代码语义正确'}

    def review_changes(self):
        """Read-only review shared by explicit tools and the final stop hook."""
        snapshot = self.backend.snapshot
        current = self.files()
        original = snapshot.manifest['files']
        changes, risks = [], []
        for name in sorted(original.keys() | current.keys()):
            data = current.get(name)
            if data is not None and hashlib.sha256(data).hexdigest() == original.get(name):
                continue
            changes.append(name)
            if name.endswith('.py'):
                before = file_bytes(self.backend.patches.baseline, name, self.backend.patches.max_file_bytes) if name in original else None
                risks.extend(python_edit_warnings(name, before, data))
        return {'changed_files': changes, 'risks': risks}


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
            report = self.backend.verification.assess()
            if report['status'] != 'not_applicable':
                report.update(self.backend.verification.review_changes())
                report['issues'].extend(report['risks'])
                if report['issues']:
                    report['status'] = 'needs_attention'
        except (OSError, ValueError, RuntimeError) as exc:
            report = {'status': 'needs_attention', 'issues': [f'证据核对失败：{type(exc).__name__}'],
                      'evidence_ids': []}
        feedback = (report['issues'] and not state.get('completion_feedback_count', 0)
                    and state.get('run_model_call_count', 0) < self.model_call_limit)
        record = {**report, 'feedback_sent': bool(feedback)}
        with (self.backend.snapshot.session / 'completion-checks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if feedback:
            return {'messages': [HumanMessage(content='程序收尾核对（最多反馈一次）：\n' +
                     '\n'.join(report['issues']) +
                     '\n有必要且预算允许时，使用 run_verification 补充验证；否则明确说明未完成或未验证。'
                     '不要为了消除提示而删除测试或缩小需求。未识别的测试输出仍是未知。')],
                    'completion_feedback_count': 1, 'jump_to': 'model'}
        reply = last.model_copy(deep=True)
        reply.response_metadata['completion_check'] = report
        if report['status'] != 'not_applicable':
            label = ('验证证据未通过核对：' + '；'.join(report['issues']) if report['issues'] else
                     '已记录验证与当前代码一致；不代表需求覆盖完整。')
            # Keep provider reasoning/tool blocks and usage intact; append only visible text.
            suffix = '\n\n[程序核对] ' + label
            if isinstance(reply.content, str):
                reply.content += suffix
            else:
                reply.content.append({'type': 'text', 'text': suffix})
        return {'messages': [reply]}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state, runtime):
        return self.review(state)

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state, runtime):
        return await asyncio.to_thread(self.review, state)
