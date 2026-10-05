"""Full middleware topology and actionable, strict evidence-node errors."""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from test_delivery_completion import receipt
from cli_rb.runtime.agent import build_agent, astream_turn
from cli_rb.runtime.persistence import SessionStore
from cli_rb.workspace.tools import make_coding_tools

backend = controlled_backend


@pytest.mark.parametrize('mode', ['legacy_limit', 'finish', 'model_limit', 'tool_limit'])
def test_full_persistent_coding_graph_reaches_real_budget(backend, mode):
    async def run():
        responses = [call('read_file', path='app.py') for _ in range(26)]
        if mode in {'legacy_limit', 'finish'}:
            responses = responses[:24] + [AIMessage(content='Read-only inspection complete.')]
        # Each scripted call needs a distinct ID just like actual provider calls.
        for i, response in enumerate(responses):
            if response.tool_calls:
                response.tool_calls[0]['id'] = f'read-{i}'
        model = ScriptedModel(replies=responses)
        session = backend.snapshot.session
        with SessionStore(session, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(session/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, coding=True, harness='full', session=store, checkpointer=saver,
                                    model_call_limit=25, tool_call_limit=21 if mode == 'tool_limit' else 50)
                if mode == 'legacy_limit':
                    agent.graph_step_limit = 160
                async def consume():
                    return [e async for e in astream_turn(agent, [HumanMessage(content='Inspect only')])]
                if mode == 'finish':
                    events = await consume()
                    assert events[-1].text == 'Read-only inspection complete.'
                    assert len(model.seen) == 25
                else:
                    expected = {'legacy_limit': GraphRecursionError,
                                'model_limit': ModelCallLimitExceededError,
                                'tool_limit': ToolCallLimitExceededError}[mode]
                    with pytest.raises(expected):
                        await consume()
                    assert len(model.seen) == (20 if mode == 'legacy_limit' else 25 if mode == 'model_limit' else 22)
    asyncio.run(run())


def test_typo_feedback_allows_exact_correction_without_rerunning_tests(backend):
    old = receipt(backend, {'old_name': 'failed'}, code=1)
    correct = 'junit:test_app::test_whitespace_with_hyphen'
    new = receipt(backend, {'test_whitespace_with_hyphen': 'passed'})
    tool = next(t for t in make_coding_tools(backend) if t.name == 'replace_verification')
    args = {'old_id': old['id'], 'new_id': new['id'],
            'old_nodes': ['junit:test_app::old_name'],
            'new_nodes': [correct.replace('with_hyphen', 'with-hyphen')],
            'reason': 'Migrated test retains the normalization assertion.'}
    error = tool.invoke(args)
    assert 'not_found' in error and 'with-hyphen' in error and correct in error
    assert not backend.verification.reconciliation_path.exists()
    assert backend.verification.assess()['issues']
    args['new_nodes'] = [correct]
    assert json.loads(tool.invoke(args))['new_nodes'] == [correct]
    assert not backend.verification.assess()['issues']
    assert len(backend.verification.records()) == 2


def test_nonpassing_nodes_are_not_suggested_as_replacements(backend):
    old = receipt(backend, {'old_name': 'failed'}, code=1)
    new = receipt(backend, {'known_bug': 'xfailed', 'current': 'passed'})
    with pytest.raises(ValueError) as error:
        backend.verification.replace_verification(old['id'], new['id'], ['junit:test_app::old_name'],
                                                 ['junit:test_app::known_bug'], 'Replace test')
    details = json.loads(str(error.value).split('Invalid test nodes: ')[1].split('. Copy exact IDs')[0])
    assert details[0]['observed_status'] == 'xfailed'
    assert details[0]['candidates'] == ['junit:test_app::current']
    assert not backend.verification.reconciliation_path.exists()
