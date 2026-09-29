import json

import pytest

from test_verification import backend as controlled_backend, success
from cli_rb.quality.verification import VerificationLedger
from cli_rb.workspace.tools import make_coding_tools

backend = controlled_backend


def register(backend):
    backend.verification.register_behavior('value', 'value is two', 'assert value == 2', ['app.py'])


def run(backend, result):
    backend.outcomes = [result]
    return json.loads(backend.run_verification('pytest', 'value behavior'))


def mixed(status='skipped'):
    marker = '<skipped/>' if status == 'skipped' else '<failure message="wrong"/>'
    return {'exit_code': 0 if status == 'skipped' else 1, 'output': '',
            'junit_xml': '<testsuite tests="2"><testcase classname="test_a" name="value"/>'
                         f'<testcase classname="test_a" name="other">{marker}</testcase></testsuite>'}


@pytest.mark.parametrize('other_status', ['skipped', 'failed'])
def test_selected_passing_nodes_do_not_hide_unselected_results(backend, other_status):
    register(backend)
    record = run(backend, mixed(other_status))
    tool = next(t for t in make_coding_tools(backend) if t.name == 'bind_verification')
    result = tool.invoke({'behavior_id': 'value', 'evidence_id': record['id'],
                         'test_nodes': ['junit:test_a::value'], 'assertion': 'value equals two'})
    assert 'Binding saved' in result
    report = VerificationLedger(backend).assess()
    assert report['behaviors'][0]['status'] == 'evidence_current'
    assert report['status'] == ('evidence_current' if other_status == 'skipped' else 'needs_attention')
    assert report['evidence'][0]['test_evidence']['counts'][other_status] == 1
    assert len(backend.verification.records()) == 1


def test_selected_skip_and_later_failure_invalidate_binding(backend):
    register(backend)
    record = run(backend, mixed())
    ledger = backend.verification
    ledger.bind_verification('value', record['id'], ['junit:test_a::other'], 'other assertion')
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'
    ledger.bind_verification('value', record['id'], ['junit:test_a::value'], 'value assertion')
    assert ledger.assess()['behaviors'][0]['status'] == 'evidence_current'
    run(backend, {'exit_code': 1, 'output': '', 'junit_xml':
                 '<testsuite tests="1"><testcase classname="test_a" name="value"><failure/></testcase></testsuite>'})
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'


def test_unknown_and_stale_nodes_cannot_be_bound(backend):
    register(backend)
    record = run(backend, success())
    ledger = backend.verification
    for nodes in [[], ['not-executed']]:
        with pytest.raises(ValueError, match='exact nodes'):
            ledger.bind_verification('value', record['id'], nodes, 'value assertion')
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    with pytest.raises(ValueError, match='stale'):
        ledger.bind_verification('value', record['id'], list(record['test_evidence']['nodes']), 'value assertion')
    assert not ledger.bindings()


def test_changed_requirement_can_reuse_actual_current_assertions_without_execution(backend):
    register(backend)
    record = run(backend, success())
    ledger = backend.verification
    nodes = list(record['test_evidence']['nodes'])
    ledger.bind_verification('value', record['id'], nodes, 'value assertion')
    ledger.register_behavior('value', 'value is two', 'the existing assertion also checks its type', ['app.py'])
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'
    ledger.bind_verification('value', record['id'], nodes, 'existing selected assertions check type and value')
    assert ledger.assess()['behaviors'][0]['status'] == 'evidence_current'
    assert len(ledger.records()) == 1
