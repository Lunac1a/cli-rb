import json
import pytest
from test_verification import backend as controlled_backend
from cli_rb.quality.evidence_lifecycle import startup_failure, resolved_startups


backend = controlled_backend

def test_recovered_startup_preserves_raw_failure_and_real_assertion_failure(backend):
    ledger = backend.verification
    ledger.register_behavior('value', 'value works', 'assert value', ['app.py'])
    backend.outcomes = [{'exit_code': 1, 'output': '/env/bin/python: No module named pytest'}]
    old = json.loads(backend.run_verification('python -m pytest tests/test_value.py', 'value', behavior_ids=['value'], assertion='assert value'))
    backend.outcomes = [{'exit_code': 0, 'output': 'Ran 2 tests in 0.1s\nOK'}]
    json.loads(backend.run_verification('python tests/runtests.py test_value.module', 'different'))
    assert old['id'] in ledger.assess()['evidence_ids']
    backend.outcomes = [{'exit_code': 0, 'output': 'Ran 2 tests in 0.1s\nOK'}]
    json.loads(backend.run_verification('python -m pytest tests/test_value.py', 'value', behavior_ids=['value'], assertion='assert value'))
    report = ledger.assess()
    assert old['id'] not in report['evidence_ids']
    assert any(x['basis'] == 'runner_startup_recovered' for x in report['replacements'])
    assert ledger.records()[0]['outcome'] == 'failed'
    backend.outcomes = [{'exit_code': 1, 'output': '1 failed'}]
    failure = json.loads(backend.run_verification('python -m pytest tests/other.py', 'value'))
    assert failure['id'] in ledger.assess()['evidence_ids']
    assert ledger.assess()['issues']


@pytest.mark.parametrize('command,output,code,timed', [
 ('python -m pytest tests/a.py -k one', '/bin/python: No module named pytest', 1, False),
 ('python -m pytest tests/a.py', 'Traceback\nModuleNotFoundError: No module named pytest', 1, False),
 ('python -m pytest tests/a.py', '/bin/python: No module named pytest', 1, True),
 ('python -m pytest tests/a.py', '/bin/python: No module named pytest', 0, False),
 ('python -m pytest tests/a.py', '1 failed', 1, False),
])
def test_ambiguous_or_actual_failures_are_not_startup_failures(command, output, code, timed):
    assert startup_failure(command, {'output': output, 'exit_code': code, 'timed_out': timed}) is None


def receipts():
    old = {'id':'old', 'kind':'test', 'command':'python -m pytest tests/forms/test_json.py', 'before':'old', 'after':'old', 'outcome':'failed', 'startup_failure':{'target':'forms.test_json'}}
    new = {'id':'new', 'kind':'test', 'command':'python tests/runtests.py forms.test_json', 'before':'current', 'after':'current', 'outcome':'passed', 'exit_code':0, 'test_evidence':{'counts':{'passed':2,'failed':0,'errors':0}}}
    return old,new


@pytest.mark.parametrize('problem', ['stale','filtered','different','failed','timeout','empty','reproduce'])
def test_no_retirement_without_current_equivalent_success(problem):
    old,new=receipts()
    if problem=='stale':
        new['before']='old'
    if problem=='filtered':
        new['command']+=' -k some'
    if problem=='different':
        new['command']='python tests/runtests.py forms.other'
    if problem=='failed':
        new['outcome']='failed'
    if problem=='timeout':
        new['timed_out']=True
    if problem=='empty':
        new['test_evidence']['counts']['passed']=0
    if problem=='reproduce':
        old['purpose']='reproduce'
    active,links=resolved_startups([old,new],'current')
    assert old in active and not links


def test_same_module_different_runner_recovers_but_later_code_edit_reopens():
    old,new=receipts()
    active,links=resolved_startups([old,new],'current')
    assert active==[new] and links[0]['basis']=='runner_startup_recovered'
    assert resolved_startups([old,new],'new-edit')[0]==[old,new]
