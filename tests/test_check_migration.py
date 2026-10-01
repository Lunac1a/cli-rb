import json
import pytest
from test_verification import backend as controlled_backend
from test_delivery_completion import receipt

backend = controlled_backend


def check(backend, code=0):
    backend.outcomes = [{'exit_code': code, 'timed_out': False, 'output': 'assertion' if code else ''}]
    return json.loads(backend.verification.run('python check.py', 'value', 10, kind='check',
                                             behavior_ids=['value'], assertion='value is 2'))


def setup(backend):
    backend.verification.register_behavior('value', 'value is 2', 'assert value == 2', ['app.py'])


def test_stale_success_is_history_but_current_evidence_is_required(backend):
    setup(backend)
    old = check(backend)
    (backend.snapshot.workspace/'app.py').write_text('value = 2\n')
    report = backend.verification.assess()
    assert report['issues'] and not any(old['id'] in i for i in report['issues'])
    new = receipt(backend, {'value': 'passed'})
    backend.verification.bind_verification('value', new['id'], ['junit:test_app::value'], 'value is 2')
    assert not backend.verification.assess()['issues']


@pytest.mark.parametrize('case', ['valid', 'mismatch', 'stale', 'timeout', 'new_failure'])
def test_failed_check_requires_explicit_current_matching_test(backend, case):
    setup(backend)
    old = check(backend, 1)
    (backend.snapshot.workspace/'app.py').write_text('value = 2\n')
    new = receipt(backend, {'value': 'passed'})
    records = backend.verification.records()
    records[-1]['behavior_contracts'] = old['behavior_contracts'] if case != 'mismatch' else {}
    if case == 'timeout':
        records[0]['timed_out'] = True
    if case == 'new_failure':
        records[-1]['outcome'] = 'failed'
    backend.verification.path.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    if case == 'stale':
        (backend.snapshot.workspace/'app.py').write_text('value = 3\n')
    assert any(old['id'] in i for i in backend.verification.assess()['issues'])
    args = (old['id'], new['id'], [], ['junit:test_app::value'], 'Test preserves the old value assertion.')
    if case != 'valid':
        with pytest.raises(ValueError):
            backend.verification.replace_verification(*args)
        return
    backend.verification.replace_verification(*args)
    assert not backend.verification.assess()['issues']
    assert len(backend.verification.records()) == 2
    (backend.snapshot.workspace/'app.py').write_text('value = 3\n')
    assert any(old['id'] in i for i in backend.verification.assess()['issues'])
