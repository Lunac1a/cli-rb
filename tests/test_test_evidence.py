import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from cli_rb.quality.test_evidence import parse_test_evidence
from cli_rb.quality.verification import VerificationLedger, parse_test_outcome
from cli_rb.runtime.agent import build_agent, stream_turn

backend = controlled_backend


def receipt(output, code=0, timed=False):
    return {'output': output, 'exit_code': code, 'timed_out': timed}


def run(backend, output, scope='same scope', code=0, **kwargs):
    backend.outcomes = [receipt(output, code)]
    return json.loads(backend.run_verification('pytest -v', scope, **kwargs))


@pytest.mark.parametrize('output,counts,total', [
    ('Ran 11 tests in 0.1s\n\nOK (skipped=1)', {'passed': 10, 'skipped': 1}, 11),
    ('==== 4 passed in 1s ====', {'passed': 4}, 4),
    ('1 passed, 2 skipped, 3 deselected, 1 warning in 1s', {'passed': 1, 'skipped': 2}, 3),
    ('Ran 4 tests in 1s\nFAILED (failures=1, errors=1)', {'failed': 1, 'errors': 1}, 4),
    ('Ran 4 tests in 1s\nOK (skipped=1, expected failures=1)', {'passed': 2, 'skipped': 1, 'xfailed': 1}, 4),
])
def test_reported_counts(output, counts, total):
    parsed = parse_test_evidence(receipt(output))
    assert parsed['total'] == total
    assert {key: value for key, value in parsed['counts'].items() if value} == counts


@pytest.mark.parametrize('output', ['hello 4 passed example', '3 passed\n4 passed', 'no tests ran'])
def test_no_invented_counts_from_ambiguous_or_unrecognized_output(output):
    assert parse_test_evidence(receipt(output))['counts'] is None
    assert parse_test_outcome(receipt(output)) == 'unknown'


def test_partial_rerun_keeps_uncovered_failure_then_full_rerun_replaces_it(backend):
    first = run(backend, 'test_a.py::a PASSED\ntest_a.py::b FAILED\n1 passed, 1 failed', code=1)
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    second = run(backend, 'test_a.py::a PASSED [100%]\n1 passed', scope='different words')
    report = backend.verification.assess()
    assert report['status'] == 'needs_attention'
    assert report['evidence'][0]['remaining_nodes'] == {'test_a.py::b': 'failed'}
    assert second['replacements'] == [{'old_id': first['id'], 'new_id': second['id'], 'nodes': ['test_a.py::a']}]
    third = run(backend, 'test_a.py::a PASSED\ntest_a.py::b PASSED\n2 passed', scope='combined')
    restored = VerificationLedger(backend).assess()
    assert restored['status'] == 'evidence_current'
    assert restored['evidence_ids'] == [third['id']]
    assert len(restored['replacements']) == 3
    assert len(backend.verification.records()) == 3


def test_same_scope_and_command_do_not_hide_unrelated_or_unidentified_failure(backend):
    first = run(backend, 'test_a.py::a FAILED\n1 failed', code=1)
    opaque = run(backend, 'ImportError: dependency missing', code=1)
    run(backend, 'test_a.py::b PASSED\n1 passed')
    report = backend.verification.assess()
    assert {first['id'], opaque['id']} <= set(report['evidence_ids'])
    assert report['status'] == 'needs_attention'


@pytest.mark.parametrize('output,code,timed', [
    ('test_a.py::a PASSED\n2 passed', 0, False),
    ('test_a.py::a PASSED\n1 passed', 1, False),
    ('test_a.py::a PASSED\n1 passed', 0, True),
    ('test_a.py::a PASSED\ntest_a.py::a ERROR\n1 passed, 1 error', 1, False),
])
def test_incomplete_or_aborted_receipts_cannot_replace(output, code, timed):
    assert not parse_test_evidence(receipt(output, code, timed))['nodes_complete']


def test_skipping_failed_node_does_not_become_pass(backend):
    run(backend, 'test_a.py::a FAILED\n1 failed', code=1)
    run(backend, 'test_a.py::a SKIPPED [100%]\n1 skipped')
    assert backend.verification.assess()['status'] == 'needs_attention'


def test_check_replacement_requires_identical_command_and_kind(backend):
    backend.outcomes = [receipt('bad', 1), receipt(''), receipt('')]
    first = json.loads(backend.run_verification('lint app.py', 'lint', kind='check'))
    backend.run_verification('lint other.py', 'lint', kind='check')
    assert first['id'] in backend.verification.assess()['evidence_ids']
    backend.run_verification('lint app.py', 'different scope', kind='check')
    assert backend.verification.assess()['status'] == 'evidence_current'


def test_final_program_summary_uses_counts_without_extra_model_calls(backend):
    backend.outcomes = [receipt('Ran 11 tests in 0.1s\nOK (skipped=1)')]
    model = ScriptedModel(replies=[call('run_verification', command='python tests.py', scope='checks'),
                                  AIMessage(content='已完成修改。')])
    events = list(stream_turn(build_agent(model, backend, coding=True), [HumanMessage(content='验证')]))
    assert len(model.seen) == 2
    assert '通过 10、跳过 1' in events[-1].text
    assert '实际节点未完整识别' in events[-1].text
    assert events[-1].response_metadata['completion_check']['evidence'][0]['test_evidence']['total'] == 11
