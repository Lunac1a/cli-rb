"""Opt-in reminder checks real edits, bounded continuation and normal chat behavior."""
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call, success
from cli_rb.runtime.agent import build_agent, stream_turn

backend = controlled_backend


def test_implementation_reminder_continues_to_real_edit_and_verification(backend):
    backend.outcomes = [success()]
    model = ScriptedModel(replies=[AIMessage(content='Seems infeasible; no changes.'),
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
        call('run_verification', command='pytest', scope='changed value'),
        AIMessage(content='Implemented and checked.')])
    agent = build_agent(model, backend, coding=True, implementation_task=True, model_call_limit=6)
    events = list(stream_turn(agent, [HumanMessage(content='Implement value two')]))
    assert events[-1].text == 'Implemented and checked.'
    assert (backend.snapshot.workspace / 'app.py').read_text() == 'value = 2\n'
    assert backend.verification.records()[0]['before'] == backend.verification.fingerprint()
    assert backend.verification.records()[0]['outcome'] == 'passed'
    assert any('Implementation-attempt check' in message.text for message in model.seen[1])
    records = (backend.snapshot.session / 'implementation-feedback.jsonl').read_text().splitlines()
    assert len(records) == 1 and json.loads(records[0])['task_success'] is None
    assert agent.requests.calls == 4


def test_implementation_reminder_does_not_force_unbounded_attempts(backend):
    model = ScriptedModel(replies=[AIMessage(content='No change.'), AIMessage(content='Still incomplete.')])
    agent = build_agent(model, backend, coding=True, implementation_task=True, model_call_limit=5)
    events = list(stream_turn(agent, [HumanMessage(content='Implement a change')]))
    assert events[-1].text == 'Still incomplete.' and len(model.seen) == 2
    assert backend.patches.diff() == ''


@pytest.mark.parametrize('enabled,limit', [(False, 5), (True, 1)])
def test_explanation_and_last_budget_reply_are_not_redirected(backend, enabled, limit):
    model = ScriptedModel(replies=[AIMessage(content='No changes; explanation only.')])
    agent = build_agent(model, backend, coding=True, implementation_task=enabled, model_call_limit=limit)
    events = list(stream_turn(agent, [HumanMessage(content='Explain the existing code')]))
    assert events[-1].text == 'No changes; explanation only.' and len(model.seen) == 1
    assert not (backend.snapshot.session / 'implementation-feedback.jsonl').exists()
