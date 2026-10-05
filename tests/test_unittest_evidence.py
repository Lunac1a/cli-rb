import json
import pytest
from test_verification import backend as controlled_backend
from cli_rb.quality.test_evidence import parse_test_evidence
from cli_rb.workspace.container_worker import django_verbose_command


backend = controlled_backend

def result(text, code=0):
    return {'output':text,'exit_code':code,'timed_out':False}


def test_verbose_docstrings_and_skips_have_exact_identities():
    d=parse_test_evidence(result("test_a (pkg.Case) ... ok\ntest_b (pkg.Case)\nA description ... ok\ntest_c (pkg.Case) ... skipped 'platform'\nRan 3 tests in 0.1s\nOK (skipped=1)"))
    assert d['nodes_complete']
    assert d['nodes']=={'unittest:pkg.Case::test_a':'passed','unittest:pkg.Case::test_b':'passed','unittest:pkg.Case::test_c':'skipped'}


@pytest.mark.parametrize('text',[
 '...\nRan 3 tests in 0.1s\nOK',
 'test_a (pkg.Case) ... ok\nRan 2 tests in 0.1s\nOK',
 'test_a (pkg.Case) ... ok\ntest_a (pkg.Case) ... ok\nRan 2 tests in 0.1s\nOK',
 'test_a (pkg.Case) ...\n  test_a (pkg.Case) (x=1) ... FAIL\nRan 1 test in 0.1s\nFAILED (failures=1)',
])
def test_missing_duplicate_or_subtest_receipts_remain_incomplete(text):
    assert not parse_test_evidence(result(text,1 if 'FAILED' in text else 0))['nodes_complete']


def test_later_actual_nodes_retire_only_covered_old_nodes(backend):
    backend.outcomes=[result('test_a (pkg.Case) ... ok\ntest_b (pkg.Case) ... ok\nRan 2 tests in 0.1s\nOK')]
    old=json.loads(backend.run_verification('python tests/runtests.py pkg', 'suite'))
    (backend.snapshot.workspace/'app.py').write_text('value = 2\n')
    backend.outcomes=[result('test_a (pkg.Case) ... ok\nRan 1 test in 0.1s\nOK')]
    backend.run_verification('python tests/runtests.py pkg.Case.test_a','subset')
    report=backend.verification.assess()
    remaining=next(x for x in report['evidence'] if x['id']==old['id'])
    assert remaining['remaining_nodes']=={'unittest:pkg.Case::test_b':'passed'}
    assert report['issues']
    backend.outcomes=[result('test_b (pkg.Case) ... ok\nRan 1 test in 0.1s\nOK')]
    backend.run_verification('python tests/runtests.py pkg.Case.test_b','other')
    assert not backend.verification.assess()['issues']


@pytest.mark.parametrize('command,allowed',[
 ('python tests/runtests.py forms.test_json',True),
 ('python3 tests/runtests.py forms.test_json another',True),
 ('python tests/runtests.py forms.test_json --verbosity=0',False),
 ('python tests/runtests.py forms.test_json; echo OK',False),
 ('python -m pytest tests/test_a.py',False),
])
def test_only_plain_django_test_commands_receive_reporting_option(command,allowed):
    assert bool(django_verbose_command(command)) is allowed
