import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, behavior_plan, success, call
from cli_rb.quality.test_evidence import parse_test_evidence
from cli_rb.quality.verification import VerificationLedger
from cli_rb.runtime.agent import build_agent, stream_turn
from cli_rb.workspace.container_worker import pytest_report_path

backend = controlled_backend


def plan(ledger, key='value', assertion='assert value == 2', paths=None, revise=False):
    ledger.record_plan([{'content': 'behavior', 'status': 'completed', 'revise_verification': revise, 'verification': {
        'id': key, 'expected': 'value is two', 'assertion': assertion, 'affected_paths': ['app.py'] if paths is None else paths}}])


def test_unknown_paths_are_valid_initial_plan_but_not_verified_completion(backend):
    plan(backend.verification, paths=[])
    run(backend, success())
    report = backend.verification.assess(require_behaviors=True)
    assert report['completion_status'] == 'partial_unverified'
    assert any('Affected paths still need investigation' in issue for issue in report['issues'])


def test_empty_required_plan_field_returns_tool_error_without_aborting(backend):
    invalid = call('register_behavior', behavior_id='', expected='behavior', assertion='check it', affected_paths=[])
    model = ScriptedModel(replies=[invalid, AIMessage(content='计划尚未登记，未修改代码')])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='调查')]))
    assert len(model.seen) == 2
    assert any('工具错误' in event.text for event in events)
    assert events[-1].text == '计划尚未登记，未修改代码'
    assert not backend.verification.plans()


def run(backend, result, purpose='verify', key='value', **kwargs):
    backend.outcomes = [result]
    return json.loads(backend.run_verification('python -m pytest -q', 'behavior',
        behavior_ids=[key], assertion='assert value == 2', purpose=purpose, **kwargs))


def test_junit_beats_nested_stdout_and_counts_skips():
    result = {'exit_code': 0, 'output': '999 passed\n1 failed', 'junit_xml': '''<testsuites>
      <testsuite tests="3"><testcase classname="test_a" name="one"/>
      <testcase classname="test_a" name="two"><skipped/></testcase>
      <testcase classname="test_a" name="three"><skipped type="pytest.xfail"/></testcase></testsuite></testsuites>'''}
    parsed = parse_test_evidence(result)
    assert parsed['source'] == 'junit' and parsed['nodes_complete']
    assert parsed['counts']['passed'] == parsed['counts']['skipped'] == parsed['counts']['xfailed'] == 1


@pytest.mark.parametrize('xml', ['<broken', '<foo/>', '<testsuite tests="2"><testcase name="one"/></testsuite>'])
def test_bad_or_incomplete_report_never_has_complete_nodes(xml):
    assert not parse_test_evidence({'exit_code': 0, 'output': '', 'junit_xml': xml})['nodes_complete']


@pytest.mark.parametrize('command', ['pytest -q tests/test_a.py', 'python -m pytest -q', 'python3 -m pytest -k "one or two"'])
def test_simple_commands_get_fresh_report_paths(command):
    path = pytest_report_path(command)
    assert path and not path.exists()


@pytest.mark.parametrize('command', ['pytest | head', 'cd tests && pytest', 'pytest --junitxml=out.xml', 'echo pytest', 'python - <<EOF\npytest\nEOF', 'pytest -- test_a.py', 'pytest # comment'])
def test_complex_commands_are_not_rewritten(command):
    assert pytest_report_path(command) is None


def test_completed_todo_without_current_bound_evidence_is_unverified(backend):
    plan(backend.verification)
    backend.outcomes = [success()]
    backend.run_verification('pytest', 'unrelated pass')
    report = backend.verification.assess(require_behaviors=True)
    assert report['completion_status'] == 'partial_unverified'
    assert report['behaviors'][0]['status'] == 'unverified'
    backend.verification.record_plan([])
    assert backend.verification.assess()['behaviors'][0]['status'] == 'unverified'


def test_reproduction_link_requires_matching_behavior_and_real_expected_failure(backend):
    plan(backend.verification)
    old = run(backend, {'exit_code': 1, 'output': 'AssertionError: wrong value'},
              purpose='reproduce', expected_failure='AssertionError')
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    new = run(backend, success())
    report = VerificationLedger(backend).assess(require_behaviors=True)
    assert report['completion_status'] == 'declared_behaviors_evidenced'
    assert report['evidence_ids'] == [new['id']]
    assert report['replacements'][0]['old_id'] == old['id']
    assert len(backend.verification.records()) == 2


def test_unexpected_reproduction_failure_is_not_erased(backend):
    plan(backend.verification)
    old = run(backend, {'exit_code': 1, 'output': 'ImportError'}, purpose='reproduce', expected_failure='AssertionError')
    run(backend, success())
    assert old['id'] in backend.verification.assess()['evidence_ids']
    assert backend.verification.assess()['status'] == 'needs_attention'


def test_contract_revision_or_code_edit_invalidates_evidence(backend):
    plan(backend.verification)
    run(backend, success())
    assert backend.verification.assess()['behaviors'][0]['status'] == 'evidence_current'
    plan(backend.verification, assertion='check a different output', revise=True)
    assert backend.verification.assess()['behaviors'][0]['status'] == 'unverified'
    run(backend, success())
    (backend.snapshot.workspace / 'app.py').write_text('value = 3\n')
    assert backend.verification.assess()['completion_status'] == 'partial_unverified'


def test_progress_reference_preserves_binding_and_accidental_redefinition_is_recoverable(backend):
    plan(backend.verification)
    run(backend, success())
    original = backend.verification.requirements()
    with pytest.raises(ValueError, match='revise_verification'):
        plan(backend.verification, assertion='paraphrased completion text')
    saved = backend.verification.record_plan([{'content': 'checked', 'status': 'completed',
                                              'verification': {'id': 'value'}}])
    assert saved[0]['verification'] == original['value']
    assert backend.verification.assess()['behaviors'][0]['status'] == 'evidence_current'
    assert len(backend.verification.records()) == 1


def test_duplicate_behavior_ids_do_not_silently_overwrite_contracts(backend):
    plan(backend.verification)
    count = len(backend.verification.plans())
    with pytest.raises(ValueError, match='only once'):
        backend.verification.record_plan([{'verification': {'id': 'value'}}] * 2)
    assert len(backend.verification.plans()) == count


def test_progress_and_exact_node_binding_complete_without_reexecution(backend, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('cli_rb.quality.verification.uuid4', lambda: SimpleNamespace(hex='bound'))
    backend.outcomes = [success()]
    model = ScriptedModel(replies=[
        call('write_todos', todos=[{'content': 'Investigate, edit, verify', 'status': 'in_progress'}]),
        behavior_plan(),
        call('run_verification', command='pytest', scope='value'),
        call('bind_verification', behavior_id='value', evidence_id='verification-bound',
             test_nodes=['tests/test_app.py::test_0'], assertion='checks value equals two'),
        call('write_todos', todos=[{'content': 'checked', 'status': 'completed'}]),
        AIMessage(content='检查结束'),
    ])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='检查')]))
    assert events[-1].response_metadata['completion_check']['behaviors'][0]['status'] == 'evidence_current'
    assert len(backend.verification.records()) == 1
    assert len(backend.verification.plans()) == 1
    assert len(backend.verification.bindings()) == 1
    assert 'affected_paths' in model.seen[-1][0].text


def test_review_shares_scan_but_next_review_detects_external_edits(backend, monkeypatch):
    ledger = backend.verification
    plan(ledger)
    run(backend, success())
    original = ledger.files
    scans = []
    def counted():
        scans.append(True)
        return original()
    monkeypatch.setattr(ledger, 'files', counted)
    assert ledger.review()['status'] == 'evidence_current'
    assert len(scans) == 1
    (backend.snapshot.workspace / 'app.py').write_text('value = 9\n')
    assert ledger.review()['status'] == 'needs_attention'
    assert len(scans) == 2


def test_other_behavior_does_not_resolve_reproduction(backend):
    plan(backend.verification)
    old = run(backend, {'exit_code': 1, 'output': 'AssertionError'}, purpose='reproduce', expected_failure='AssertionError')
    plan(backend.verification, key='other')
    run(backend, success(), key='other')
    assert old['id'] in backend.verification.assess()['evidence_ids']


def test_program_status_leads_final_reply_even_when_model_claims_done(backend):
    model = ScriptedModel(replies=[behavior_plan(), AIMessage(content='全部完成'), AIMessage(content='仍然全部完成')])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='检查')]))
    assert len(model.seen) == 3
    assert events[-1].text.startswith('[完成状态] 部分完成')
    assert events[-1].response_metadata['completion_check']['behaviors'][0]['status'] == 'unverified'
