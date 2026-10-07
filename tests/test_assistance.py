"""Assistance facts must reflect actual execution, without registration gates."""
import asyncio
import json
import subprocess
import pytest
from pydantic import ValidationError
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call, success
from cli_rb.runtime.agent import build_agent, stream_turn, astream_turn
from cli_rb.runtime.persistence import SessionStore

backend = controlled_backend


def test_verification_timeout_schema_matches_execution_boundary(backend):
    agent = build_agent(ScriptedModel(replies=[]), backend, coding=True)
    from cli_rb.runtime.assistance import assisted_tools
    verification = next(t for t in assisted_tools(backend, [], agent.assistance)
                        if t.name == 'run_verification')
    schema = verification.args_schema
    for seconds in (1, 60):
        assert schema.model_validate({'command': 'pytest', 'timeout_seconds': seconds}).timeout_seconds == seconds
    for seconds in (0, 61, 180):
        with pytest.raises(ValidationError):
            schema.model_validate({'command': 'pytest', 'timeout_seconds': seconds})
    bounds = schema.model_json_schema()['properties']['timeout_seconds']
    assert bounds['minimum'] == 1 and bounds['maximum'] == 60


def observations(messages):
    text = messages[0].text.split('Program observations (facts, not semantic acceptance):\n')[1]
    return json.JSONDecoder().raw_decode(text)[0]


def test_automatic_version_facts_and_minimal_verification(backend):
    backend.outcomes = [success()]
    model = ScriptedModel(replies=[call('run_verification', command='pytest'),
        call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2'),
        call('get_task_state', section='evidence'), AIMessage(content='Changed; current version unverified.')])
    agent = build_agent(model, backend, coding=True)
    events = list(stream_turn(agent, [HumanMessage(content='Change value to two')]))
    assert agent.completion is None and agent.assistance is not None
    assert 'register_behavior' not in agent.assistance.tool_names
    assert 'write_todos' not in agent.assistance.tool_names
    assert observations(model.seen[1])['evidence_counts'] == {'current_passed': 1}
    stale = observations(model.seen[2])
    assert stale['evidence_counts'] == {'historical_passed': 1}
    assert stale['changed_files'] == [{'path': 'app.py', 'change': 'modified'}]
    assert stale['budget']['model_calls_remaining_including_next'] == 48
    assert stale['task_success'] is None
    receipt = next(e for e in events if getattr(e, 'name', None) == 'get_task_state')
    assert json.loads(receipt.content)['items'][0]['current'] is False
    assert len(backend.verification.records()) == 1
    assert backend.verification.requirements() == {}
    final = json.loads((backend.snapshot.session / 'assistance-final.json').read_text())
    assert final['phase'] == 'reply_returned' and final['task_success'] is None


def test_external_edit_and_reverted_version_preserve_historical_evidence(backend):
    backend.outcomes = [success(), {'exit_code': 0, 'output': 'no tests ran'}]
    backend.run_verification('pytest', 'existing check')
    agent = build_agent(ScriptedModel(replies=[]), backend, coding=True, harness='assist')
    status = agent.assistance
    (backend.snapshot.workspace / 'app.py').write_text('value = 2\n')
    assert status.collect()['evidence'][0]['current'] is False
    (backend.snapshot.workspace / 'app.py').write_text('value = 1\n')
    assert status.collect()['evidence'][0]['current'] is True
    backend.run_verification('pytest empty', 'empty suite')
    result = json.loads(status.status_tool().invoke({'section': 'evidence', 'offset': 1, 'limit': 1}))
    assert result['items'][0]['outcome'] == 'unknown'
    assert result['items'][0]['current'] is True
    assert result['total'] == 2 and result['next_offset'] is None


def test_denied_delete_does_not_change_files_or_erase_evidence(backend):
    subprocess.run(['git', 'init', str(backend.snapshot.workspace)], check=True, capture_output=True)
    backend.snapshot = replace(backend.snapshot, mode='direct')
    backend.approve = lambda _: False
    backend.outcomes = [success()]
    backend.run_verification('pytest', 'old receipt')
    model = ScriptedModel(replies=[call('delete_file', path='app.py', expected_content='value = 1\n'),
                                    AIMessage(content='Deletion denied.')])
    agent = build_agent(model, backend, coding=True, harness='assist')
    list(stream_turn(agent, [HumanMessage(content='Delete app.py')]))
    assert (backend.snapshot.workspace / 'app.py').read_text() == 'value = 1\n'
    facts = observations(model.seen[1])
    assert facts['changed_file_count'] == 0 and facts['evidence_counts']['current_passed'] == 1


def test_context_includes_fact_card_and_persistence_does_not_certify_completion(backend):
    async def run():
        model = ScriptedModel(replies=[call('get_task_state'), AIMessage(content='Inspection only.')])
        with SessionStore(backend.snapshot.session, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(store.directory / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, coding=True, harness='assist', session=store, checkpointer=saver)
                async for _ in astream_turn(agent, [HumanMessage(content='Inspect the task state')]):
                    pass
                assert observations(model.seen[0])['budget']['context_input_estimate'] > 0
                assert observations(model.seen[1])['budget']['model_calls_remaining_including_next'] == 49
                records = [json.loads(x) for x in (store.directory / 'context-calls.jsonl').read_text().splitlines()]
                assert any(x['kind'] == 'model' and x['input_estimate'] > 0 for x in records)
                final = json.loads((store.directory / 'assistance-final.json').read_text())
                assert final['task_success'] is None and store.data['state'] == 'idle'
    asyncio.run(run())
