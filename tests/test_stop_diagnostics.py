import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from cli_rb.runtime.agent import build_agent, stream_turn, TaskTimeLimitExceeded
from cli_rb.runtime.stop_diagnostics import termination_record


backend = controlled_backend

def test_final_reply_records_stop_separately_from_incomplete_work(backend):
    model = ScriptedModel(replies=[call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'), AIMessage(content='done'), AIMessage(content='unverified')])
    list(stream_turn(build_agent(model, backend, coding=True, harness='full'), [HumanMessage(content='edit')]))
    records = [json.loads(line) for line in (backend.snapshot.session / 'stop-diagnostics.jsonl').read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record['trigger'] == 'model_final_reply'
    assert record['verification']['completion_status'] == 'partial_unverified'
    assert record['feedback']['sent'] == 1
    assert record['feedback']['new_evidence_ids_after_first_feedback'] == []
    assert record['budget']['model_calls_used'] == 3
    assert record['budget']['model_calls_remaining'] == 47
    assert record['task_success'] is None
    assert len(model.seen) == 3
    assert record['feedback']['last_decision'] == 'feedback_already_used'


def test_budget_reserved_final_is_not_model_voluntary_stop(backend):
    model = ScriptedModel(replies=[AIMessage(content='unverified')])
    list(stream_turn(build_agent(model, backend, coding=True, harness='full', model_call_limit=1), [HumanMessage(content='inspect')]))
    record = json.loads((backend.snapshot.session / 'stop-diagnostics.jsonl').read_text().splitlines()[-1])
    assert record['trigger'] == 'model_budget_final_report'
    assert record['budget']['model_calls_remaining'] == 0


def test_runtime_failure_persists_diagnostics_and_still_raises(backend):
    model = ScriptedModel(replies=[])
    with pytest.raises(IndexError):
        list(stream_turn(build_agent(model, backend, coding=True, harness='full'), [HumanMessage(content='inspect')]))
    record = json.loads((backend.snapshot.session / 'stop-diagnostics.jsonl').read_text().splitlines()[-1])
    assert record['trigger'] == 'runtime_exception'
    assert record['error_type'] == 'IndexError'


@pytest.mark.parametrize('error,trigger', [(TaskTimeLimitExceeded(1), 'time_budget'), (RuntimeError('provider failed'), 'runtime_exception')])
def test_exception_is_not_misreported_as_voluntary_stop(error, trigger):
    report = {'issues': ['failed'], 'evidence': [{'id': 'old', 'current': False, 'remaining_outcome': 'failed'}, {'id': 'new', 'current': True, 'remaining_outcome': 'failed', 'remaining_nodes': None}]}
    result = termination_record(error, True, report, elapsed=2, time_limit=1, model_calls=4, model_limit=50, tool_results=2, checks=[])
    assert result['trigger'] == trigger
    assert [x['category'] for x in result['verification']['gaps']] == ['stale_evidence', 'verification_failed']
    assert result['verification']['gaps'][1]['cause'] == 'unknown'
    assert result['budget']['seconds_remaining'] == 0
    assert result['budget']['tool_calls_remaining'] is None
