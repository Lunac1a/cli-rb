import json
import asyncio
import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedModel
from cli_rb.runtime.agent import build_agent, stream_turn
from cli_rb.runtime.agent import astream_turn
from cli_rb.runtime.persistence import SessionStore
from cli_rb.workspace.coding_backend import DockerCodingBackend
from cli_rb.workspace.container_worker import execute
from cli_rb.workspace.snapshot import create_snapshot
from cli_rb.quality.verification import VerificationLedger, parse_test_outcome
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from cli_rb.evaluation.scoring import evidence_coverage


class ControlledBackend(DockerCodingBackend):
    """Only command receipts are simulated; edits and ledger use real temporary files."""
    def _execute_call(self, action, args):
        if action != 'command':
            return execute(self.snapshot.workspace, {'action': action, 'args': args}, writable=True)
        result = self.outcomes.pop(0)
        if callable(result):
            result = result()
        name = f'command-{len(self.command_evidence)}'
        self.command_evidence.append({'log_id': name, 'command': args['command']})
        raw = json.dumps(result)
        (self.logs / (name + '.txt')).write_text(raw)
        return raw[:16000]


@pytest.fixture
def backend(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'app.py').write_text('value = 1\n')
    snapshot = create_snapshot(repo, tmp_path / 'sessions')
    backend = ControlledBackend(snapshot, 'docker', 'fake-image')
    backend.outcomes = []
    return backend


def success():
    return {'exit_code': 0, 'timed_out': False, 'output': '3 passed in 0.1s'}


def call(name, **args):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': name, 'type': 'tool_call'}])


@pytest.mark.parametrize('output,code,timed,status', [
    ('3 passed in 1s', 0, False, 'passed'),
    ('Ran 3 tests in 0.01s\n\nOK', 0, False, 'passed'),
    ('Ran 3 tests in 0.01s\n\nOK (skipped=3)', 0, False, 'unknown'),
    ('Ran 0 tests in 0.0s\nOK', 0, False, 'unknown'),
    ('no tests ran', 0, False, 'unknown'),
    ('hello', 0, False, 'unknown'),
    ('1 failed, 3 passed', 0, False, 'failed'),
    ('3 passed', 1, False, 'failed'),
    ('3 passed', 0, True, 'failed'),
])
def test_observed_results_not_exit_code_alone(output, code, timed, status):
    assert parse_test_outcome({'output': output, 'exit_code': code, 'timed_out': timed}) == status


def test_ledger_survives_reload_detects_external_edits_and_reads_full_output(backend):
    backend.outcomes = [{**success(), 'output': 'x' * 20000 + '\n3 passed in 0.1s'}]
    evidence = json.loads(backend.run_verification('pytest', 'parser'))
    assert evidence['outcome'] == 'passed'
    restored = VerificationLedger(backend)
    assert restored.assess()['status'] == 'evidence_current'
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    assert restored.assess()['status'] == 'needs_attention'
    backend.outcomes = [success()]
    backend.run_verification('pytest', 'parser')
    assert restored.assess()['status'] == 'evidence_current'


def test_command_that_changes_source_cannot_validate_its_own_final_version(backend):
    def edit_during_command():
        (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
        return success()
    backend.outcomes = [edit_during_command]
    backend.run_verification('pytest', 'parser')
    assert backend.verification.assess()['status'] == 'needs_attention'


def test_check_success_is_distinct_from_test_success(backend):
    backend.outcomes = [{'exit_code': 0, 'timed_out': False, 'output': ''}]
    record = json.loads(backend.run_verification('python -c assert', 'assertion reproduction', kind='check'))
    assert record['outcome'] == 'command_succeeded'


def test_framework_feedback_repairs_before_releasing_final_answer(backend):
    backend.outcomes = [success()]
    model = ScriptedModel(replies=[
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
        AIMessage(content='UNVERIFIED DRAFT'),
        call('run_verification', command='pytest', scope='value behavior'),
        AIMessage(content='已修改并验证。'),
    ])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='修复')]))
    assert len(model.seen) == 4
    assert not any('UNVERIFIED DRAFT' in e.text for e in events)
    assert events[-1].response_metadata['completion_check']['status'] == 'evidence_current'
    assert '[程序核对]' in events[-1].text
    assert any('程序收尾核对' in m.text for m in model.seen[2])


def test_feedback_is_bounded_and_never_hides_unresolved_evidence(backend):
    model = ScriptedModel(replies=[
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
        AIMessage(content='完成'), AIMessage(content='暂未验证'),
    ])
    result = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='修改')]))
    assert len(model.seen) == 3
    assert result[-1].response_metadata['completion_check']['status'] == 'needs_attention'
    assert '验证证据未通过' in result[-1].text


def test_no_change_question_does_not_require_tests_or_extra_calls(backend):
    model = ScriptedModel(replies=[AIMessage(content='代码说明')])
    result = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='解释')]))
    assert len(model.seen) == 1 and result[-1].text == '代码说明'


def test_final_check_respects_last_model_call(backend):
    model = ScriptedModel(replies=[
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'), AIMessage(content='完成')])
    result = list(stream_turn(build_agent(model, backend, coding=True, model_call_limit=2), [HumanMessage(content='修改')]))
    assert len(model.seen) == 2
    assert result[-1].response_metadata['completion_check']['status'] == 'needs_attention'


def test_new_turn_readonly_does_not_reopen_old_failure(backend):
    backend.outcomes = [{'exit_code': 1, 'timed_out': False, 'output': '1 failed'}]
    backend.run_verification('pytest', 'parser')
    backend.verification.begin_turn()
    assert backend.verification.assess()['status'] == 'not_applicable'


def test_navigation_and_edit_feedback(tmp_path):
    (tmp_path / 'tests').mkdir()
    p = tmp_path / 'tests/test_a.py'
    p.write_text('def test_old():\n    assert True\n')
    assert execute(tmp_path, {'action': 'directory', 'args': {'path': '.'}}) == 'tests/'
    assert execute(tmp_path, {'action': 'find', 'args': {'pattern': '*test_a.py'}}) == 'tests/test_a.py'
    assert 'test_old' in execute(tmp_path, {'action': 'search', 'args': {'path': 'tests/test_a.py', 'query': 'def'}})
    warning = execute(tmp_path, {'action': 'edit', 'args': {'path': 'tests/test_a.py',
        'old_text': 'def test_old():', 'new_text': 'def test_new():'}}, writable=True)
    assert 'test_old' in warning and '改变归属' in warning


def test_missing_evaluation_nodes_are_not_resolved_false():
    result = evidence_coverage(['target', 'regression'], {'target': 'PASSED'})
    assert not result['valid'] and result['missing'] == ['regression']
    assert evidence_coverage(['target'], {'target': 'FAILED'})['valid']
    assert not evidence_coverage(['target'], {'target': 'PASSED'}, False)['valid']


def test_persisted_final_audit_survives_reopening_and_next_question(backend):
    async def run():
        session = backend.snapshot.session
        backend.outcomes = [success()]
        with SessionStore(session, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(session/'checkpoints.sqlite')) as saver:
                model = ScriptedModel(replies=[
                    call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
                    call('run_verification', command='pytest', scope='value'), AIMessage(content='已验证')])
                agent = build_agent(model, backend, coding=True, session=store, checkpointer=saver)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='修改')])]
                assert events[-1].response_metadata['completion_check']['status'] == 'evidence_current'
        with SessionStore(session) as store:
            async with AsyncSqliteSaver.from_conn_string(str(session/'checkpoints.sqlite')) as saver:
                model = ScriptedModel(replies=[AIMessage(content='这是说明')])
                agent = build_agent(model, backend, coding=True, session=store, checkpointer=saver)
                saved = await store.history(agent)
                assert saved[-1].response_metadata['completion_check']['status'] == 'evidence_current'
                events = [e async for e in astream_turn(agent, [HumanMessage(content='只解释')])]
                assert events[-1].text == '这是说明'
    asyncio.run(run())
