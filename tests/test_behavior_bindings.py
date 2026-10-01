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


def test_declared_whole_run_needs_no_binding_or_second_execution(backend):
    register(backend)
    backend.outcomes = [success()]
    receipt = json.loads(backend.run_verification(
        'pytest', 'value', behavior_ids=['value'], assertion='value equals two'))
    report = backend.verification.assess()
    assert report['behaviors'][0]['status'] == 'evidence_current'
    assert report['behaviors'][0]['evidence_ids'] == [receipt['id']]
    assert not backend.verification.bindings()
    assert len(backend.command_evidence) == 1


def test_separate_bindings_preserve_core_and_compatibility_coverage(backend):
    register(backend)
    ledger = backend.verification
    core = run(backend, success())
    core_nodes = list(core['test_evidence']['nodes'])
    ledger.bind_verification('value', core['id'], core_nodes, 'core contract')
    other = run(backend, mixed())
    ledger.bind_verification('value', other['id'], ['junit:test_a::value'], 'compatibility')
    report = VerificationLedger(backend).assess()['behaviors'][0]
    assert report['status'] == 'evidence_current'
    assert set(report['bound_nodes']) == set(core_nodes + ['junit:test_a::value'])
    assert report['evidence_ids'] == [core['id'], other['id']]
    assert report['assertion_claims'] == ['core contract', 'compatibility']
    # A later observation of the earlier core behavior must invalidate it,
    # even though the most recent binding only selected compatibility.
    run(backend, {'exit_code': 1, 'output':
        core_nodes[0] + ' FAILED [100%]\n1 failed in 0.1s'})
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'


@pytest.mark.parametrize('result', [success(), {'exit_code': 0, 'output': 'assertions checked'}])
def test_bound_command_accepts_both_success_receipts_and_retains_later_failure(backend, result):
    register(backend)
    ledger = backend.verification
    backend.outcomes = [result]
    receipt = json.loads(backend.run_verification('python check.py', 'value', kind='check'))
    ledger.bind_verification('value', receipt['id'], [], 'value equals two')
    assert ledger.assess()['behaviors'][0]['status'] == 'evidence_current'
    backend.outcomes = [{'exit_code': 1, 'output': 'AssertionError'}]
    backend.run_verification('python check.py', 'value', kind='check')
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'
