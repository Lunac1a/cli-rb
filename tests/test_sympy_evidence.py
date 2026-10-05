"""Native summaries are execution evidence, never task or node coverage."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from test_assistance import observations
from cli_rb.quality.test_evidence import parse_test_evidence
from cli_rb.quality.verification import parse_test_outcome
from cli_rb.runtime.agent import build_agent, stream_turn

backend = controlled_backend


def receipt(summary):
    return {'output': f'======= tests finished: {summary}, in 5.84 seconds =======\n',
            'exit_code': 0, 'timed_out': False}


@pytest.mark.parametrize('summary,counts,outcome', [
    ('186 passed, 7 expected to fail', {'passed': 186, 'xfailed': 7}, 'passed'),
    ('244 passed, 5 expected to fail', {'passed': 244, 'xfailed': 5}, 'passed'),
    ('8 passed, 1 failed, 1 expected to fail', {'passed': 8, 'failed': 1, 'xfailed': 1}, 'failed'),
    ('4 passed,\n8 exceptions', {'passed': 4, 'errors': 8}, 'failed'),
    ('4 passed, 1 expected to fail but passed', {'passed': 4, 'xpassed': 1}, 'failed'),
    ('0 passed, 2 skipped', {'skipped': 2}, 'unknown'),
    ('0 passed', {}, 'unknown'),
    ('4 passed, 1 failed, 2 failed', {'passed': 4, 'failed': 3}, 'failed'),
])
def test_native_counts_and_outcomes(summary, counts, outcome):
    result = receipt(summary)
    evidence = parse_test_evidence(result)
    assert evidence['framework'] == 'sympy'
    assert {k: v for k, v in evidence['counts'].items() if v} == counts
    assert evidence['total'] == sum(counts.values())
    assert evidence['nodes'] == {} and not evidence['nodes_complete']
    assert parse_test_outcome(result) == outcome


def test_multiple_or_incomplete_runs_do_not_invent_success():
    result = receipt('3 passed')
    result['output'] *= 2
    assert parse_test_outcome(result) == 'unknown'
    assert parse_test_outcome({'output': '=== tests finished: 3 passed,', 'exit_code': 0}) == 'unknown'
    for change in ({'timed_out': True}, {'exit_code': 1}):
        assert parse_test_outcome(receipt('3 passed') | change) == 'failed'


def test_assist_exposes_success_then_invalidates_after_edit(backend):
    backend.outcomes = [receipt('186 passed, 7 expected to fail')]
    model = ScriptedModel(replies=[call('run_verification', command='python native_tests.py'),
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
        AIMessage(content='Changed; needs verification.')])
    list(stream_turn(build_agent(model, backend, coding=True, harness='assist'),
                     [HumanMessage(content='Run checks, then edit.')]))
    assert observations(model.seen[1])['evidence_counts'] == {'current_passed': 1}
    assert observations(model.seen[2])['evidence_counts'] == {'historical_passed': 1}
    assert observations(model.seen[2])['task_success'] is None
    record = backend.verification.records()[0]
    assert record['test_evidence']['counts']['xfailed'] == 7
    assert not record['test_evidence']['nodes_complete']
    final = json.loads((backend.snapshot.session / 'assistance-final.json').read_text())
    assert final['task_success'] is None
