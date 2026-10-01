"""Regression cases for generated artifacts and evidence reconciliation."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from cli_rb.quality.verification import VerificationLedger
from cli_rb.runtime.agent import build_agent, stream_turn
from cli_rb.workspace.patches import text_patch
from cli_rb.workspace.tools import make_coding_tools

backend = controlled_backend


def receipt(backend, nodes, *, code=0):
    statuses = {'passed': '', 'failed': '<failure/>', 'xfailed': '<skipped type="pytest.xfail"/>'}
    backend.outcomes = [{'exit_code': code, 'output': '', 'junit_xml':
        f'<testsuite tests="{len(nodes)}">' + ''.join(
            f'<testcase classname="test_app" name="{name}">{statuses[status]}</testcase>'
            for name, status in nodes.items()) + '</testsuite>'}]
    return json.loads(backend.run_verification('pytest tests/test_app.py', 'behavior'))


def test_test_cache_does_not_change_fingerprint_or_block_export(backend):
    root = backend.snapshot.workspace
    (root / 'app.py').write_text('value = 2\n')
    (root / 'new_module.py').write_text('value = 3\n')
    (root / 'assets').mkdir()
    (root / 'assets/data.txt').write_text('required resource\n')
    before = backend.verification.fingerprint()
    cache = root / '.hypothesis/examples'
    cache.mkdir(parents=True)
    (cache / 'example').write_bytes(b'\x00\xff')
    assert backend.verification.fingerprint() == before
    patch = backend.patches.diff()
    assert all(name in patch for name in ['app.py', 'new_module.py', 'assets/data.txt'])
    assert '.hypothesis' not in patch
    backend.patches.export()
    assert next((backend.snapshot.session / 'exports').glob('*.patch')).read_bytes() == patch.encode('utf-8')
    assert (cache / 'example').read_bytes() == b'\x00\xff'
    (root / 'assets/real.bin').write_bytes(b'\x00\xff')
    with pytest.raises(ValueError, match='二进制'):
        backend.patches.diff()
    with pytest.raises(ValueError, match='二进制'):
        text_patch({'asset.bin': b'\0a'}, {'asset.bin': b'\0b'})


def test_repeated_xfail_is_current_but_not_passing_behavior(backend):
    ledger = backend.verification
    ledger.register_behavior('value', 'works', 'assert value', ['app.py'])
    old = receipt(backend, {'works': 'passed', 'known_bug': 'xfailed'})
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    new = receipt(backend, {'works': 'passed', 'known_bug': 'xfailed'})
    ledger.bind_verification('value', new['id'], ['junit:test_app::works'], 'assert value')
    report = ledger.assess(require_behaviors=True)
    assert report['issues'] == [] and old['id'] not in report['evidence_ids']
    ledger.bind_verification('value', new['id'], ['junit:test_app::known_bug'], 'still expected to fail')
    assert ledger.assess()['behaviors'][0]['status'] == 'unverified'


def test_remaining_xfail_keeps_its_label_without_poisoning_a_passing_subset(backend):
    receipt(backend, {'works': 'passed', 'known_bug': 'xfailed'})
    receipt(backend, {'works': 'passed'})
    report = backend.verification.assess()
    assert not report['issues']
    assert report['evidence'][0]['remaining_outcome'] == 'expected_failure'


def test_xfail_cannot_erase_a_real_failure(backend):
    old = receipt(backend, {'works': 'failed'}, code=1)
    receipt(backend, {'works': 'xfailed'})
    report = backend.verification.assess()
    assert old['id'] in report['evidence_ids'] and report['issues']


def test_explicit_test_replacement_survives_reload_and_preserves_other_failures(backend):
    old = receipt(backend, {'old_name': 'failed', 'other': 'failed'}, code=1)
    new = receipt(backend, {'new_name': 'passed'})
    tool = next(t for t in make_coding_tools(backend) if t.name == 'replace_verification')
    result = tool.invoke({'old_id': old['id'], 'new_id': new['id'],
                         'old_nodes': ['junit:test_app::old_name'],
                         'new_nodes': ['junit:test_app::new_name'],
                         'reason': 'Renamed assertion still checks the same returned value.'})
    assert json.loads(result)['new_id'] == new['id']
    restored = VerificationLedger(backend)
    report = restored.assess()
    assert report['evidence'][0]['remaining_nodes'] == {'junit:test_app::other': 'failed'}
    assert report['issues'] and len(restored.records()) == 2
    receipt(backend, {'other': 'passed'})
    assert not restored.assess()['issues']
    # A later failure at the replacement node still blocks closure.
    receipt(backend, {'new_name': 'failed'}, code=1)
    assert restored.assess()['issues']
    # Editing code invalidates the replacement claim as well as the receipts.
    (backend.snapshot.workspace / 'app.py').write_text('value = 3\n')
    assert restored.assess()['evidence'][0]['remaining_nodes']['junit:test_app::old_name'] == 'failed'


@pytest.mark.parametrize('problem', ['stale', 'unobserved', 'xfail', 'process_error', 'blank_reason'])
def test_invalid_replacement_never_retires_old_failure(backend, problem):
    old = receipt(backend, {'old_name': 'failed'}, code=1)
    new = receipt(backend, {'new_name': 'xfailed' if problem == 'xfail' else 'passed'},
                  code=1 if problem == 'process_error' else 0)
    if problem == 'stale':
        (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    with pytest.raises(ValueError):
        backend.verification.replace_verification(
            old['id'], new['id'], ['junit:test_app::old_name'],
            ['junit:test_app::missing' if problem == 'unobserved' else 'junit:test_app::new_name'],
            '' if problem == 'blank_reason' else 'Renamed assertion')
    assert old['id'] in backend.verification.assess()['evidence_ids']


def test_test_migration_reminder_is_visible_without_a_feedback_loop(backend):
    root = backend.snapshot.workspace
    # Establish the same baseline surface as an existing test file.
    name, content = 'test_old.py', b'def test_value():\n    assert True\n'
    import hashlib
    backend.snapshot.manifest['files'][name] = hashlib.sha256(content).hexdigest()
    (backend.patches.baseline / name).write_bytes(content)
    backend.patches._before = None
    (root / name).write_bytes(content)
    backend.outcomes = [{'exit_code': 0, 'output': 'test_new.py::test_value PASSED\n1 passed'}]
    model = ScriptedModel(replies=[
        call('register_behavior', behavior_id='value', expected='works', assertion='value is correct', affected_paths=['app.py']),
        call('delete_file', path=name, expected_content=content.decode()),
        call('create_file', path='test_new.py', content=content.decode()),
        call('run_verification', command='pytest', scope='value', behavior_ids=['value'], assertion='value is correct'),
        AIMessage(content='Moved the test; current test passes.'),
    ])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='Move tests')]))
    assert len(model.seen) == 5
    final = events[-1]
    assert final.response_metadata['completion_check']['completion_status'] == 'declared_behaviors_evidenced'
    assert '审阅提醒' in final.text and 'test_old.py' in final.text
    (root / 'app.py').write_text('def broken(:\n')
    report = backend.verification.review()
    assert report['syntax_errors'] and report['completion_status'] == 'partial_unverified'
