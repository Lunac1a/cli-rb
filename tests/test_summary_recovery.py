import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from helpers import ScriptedModel
from test_model_recovery import FaultModel, failure
from cli_rb.runtime.agent import build_agent, astream_turn
from cli_rb.runtime.context import ContextPolicy
from cli_rb.runtime.persistence import SessionStore
from cli_rb.interface.demo import demo_backend


@pytest.mark.parametrize('characters', [4156, 4929, 16000])
def test_soft_summary_target_uses_total_request_acceptance(tmp_path, characters):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            model = ScriptedModel(replies=[AIMessage(content='x' * characters)])
            context = build_agent(model, demo_backend(), session=store,
                                  context_policy=ContextPolicy(window=32768)).context
            user = HumanMessage(content='Never delete original data.', id='u')
            old = AIMessage(content='a' * 48000)
            update = await context.abefore_model({'messages': [old, user]}, None)
            assert update['context_summary_count'] == 1
            assert user in update['messages'] and old not in update['messages']
            assert update['context_estimate'] <= context.policy.input_limit * 0.75
            assert update['context_estimate'] <= context.total_size([old, user]) / 2
            record = json.loads((tmp_path/'context-calls.jsonl').read_text())
            assert record['above_soft_target'] and record['outcome'] == 'accepted'
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['retry_summary', 'retry_main', 'limit'])
def test_summary_and_main_retries_share_original_budget(tmp_path, mode):
    async def run():
        # Three actual requests at most: two summaries + answer, or summary + HTTP failure + answer.
        outcomes = ([AIMessage(content='z' * 60000), AIMessage(content='Useful handoff'), AIMessage(content='Done')]
                    if mode == 'retry_summary' else
                    [AIMessage(content='Useful handoff'), failure(), AIMessage(content='Done')])
        model = FaultModel(replies=[], outcomes=outcomes)
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver,
                                    model_call_limit=2 if mode == 'limit' else 3,
                                    context_policy=ContextPolicy(window=32768))
                agent.requests.retry.initial_delay = 0
                await agent.graph.aupdate_state(store.config, {'messages': [AIMessage(content='a'*48000)]}, as_node=START)
                await agent.graph.aupdate_state(store.config, None, as_node=END)
                await store.finish(agent)
                async def consume():
                    return [e async for e in astream_turn(agent, [HumanMessage(content='Continue')])]
                if mode == 'limit':
                    with pytest.raises(ModelCallLimitExceededError):
                        await consume()
                    assert len(model.seen) == 2
                else:
                    events = await consume()
                    assert events[-1].text == 'Done'
                    assert len(model.seen) == agent.requests.calls == 3
                    state = await agent.graph.aget_state(store.config)
                    assert state.values['thread_model_call_count'] == 3
                    if mode == 'retry_summary':
                        assert 'Previous handoff' in model.seen[1][-1].text
                        decisions = [json.loads(s) for s in (tmp_path/'context-decisions.jsonl').read_text().splitlines()]
                        assert decisions[0]['attempts'] == 2 and decisions[0]['outcome'] == 'committed'
    asyncio.run(run())
