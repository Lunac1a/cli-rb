import asyncio
import json
from types import SimpleNamespace

import pytest
from helpers import ScriptedModel, request
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from cli_rb.runtime.agent import astream_turn, build_agent, stream_turn
from cli_rb.runtime.context import ContextLimitExceeded
from cli_rb.runtime.persistence import SessionStore
from cli_rb.workspace.tools import FixtureBackend


def backend():
    value = FixtureBackend({'README.md': 'fixture'})
    value.snapshot = SimpleNamespace(mode='isolated')
    return value


def plan_call(todos):
    return AIMessage(content='', tool_calls=[{
        'name': 'write_todos', 'args': {'todos': todos}, 'id': 'plan-1', 'type': 'tool_call',
    }])


def test_plan_survives_history_replacement_reopen_and_confirmed_recovery(tmp_path):
    todos = [{'content': '验证边界 ORCHID-731；完成条件：断言符合预期', 'status': 'in_progress'}]

    async def run():
        model = ScriptedModel(replies=[plan_call(todos), AIMessage(content='验证尚未完成')])
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend(), coding=True, harness='full', session=store, checkpointer=saver)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='调查并修复问题')])]
                assert any(isinstance(e, ToolMessage) and e.name == 'write_todos' for e in events)
                assert 'ORCHID-731' in model.seen[-1][0].text
                state = await agent.graph.aget_state(store.config)
                assert state.values['todos'] == todos
                # Replace message history as a compaction would; the todo channel remains.
                await agent.graph.aupdate_state(store.config, {'messages': [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    HumanMessage(content='调查并修复问题'), AIMessage(content='旧工作摘要，验证未完成'),
                ]})
                await store.finish(agent)
        with SessionStore(tmp_path) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
                model = ScriptedModel(replies=[request('README.md', 'pending'), AIMessage(content='仍需验证')])
                agent = build_agent(model, backend(), coding=True, harness='full', session=store, checkpointer=saver)
                await store.begin(agent)
                await agent.graph.ainvoke({'messages': [HumanMessage(content='继续调查')]},
                                         store.config, interrupt_before=['tools'])
                store.block()
                state = await agent.graph.aget_state(store.config)
                await store.recover(agent, state, store.action_fingerprints())
                assert (await agent.graph.aget_state(store.config)).values['todos'] == todos
                events = [e async for e in astream_turn(agent, [HumanMessage(content='只说明待办')])]
                assert events[-1].text == '仍需验证'
                assert 'ORCHID-731' in model.seen[-1][0].text
                assert 'writes require renewed authorization' in model.seen[-1][0].text
                assert not any(isinstance(e, ToolMessage) for e in events)
                records = [json.loads(line) for line in (tmp_path/'context-calls.jsonl').read_text().splitlines()]
                assert records[-1]['input_estimate'] > 0
    asyncio.run(run())


def test_planning_counts_against_tool_budget_and_works_without_session():
    todos = [{'content': '先定位根因', 'status': 'in_progress'}]
    model = ScriptedModel(replies=[plan_call(todos), request('README.md', 'over-budget')])
    with pytest.raises(ToolCallLimitExceededError):
        list(stream_turn(build_agent(model, backend(), coding=True, harness='full', tool_call_limit=1),
                         [HumanMessage(content='修复问题')]))
    assert '先定位根因' in model.seen[-1][0].text


def test_oversized_persisted_plan_cannot_bypass_input_budget(tmp_path):
    async def run():
        model = ScriptedModel(replies=[])
        with SessionStore(tmp_path, create=True) as store:
            agent = build_agent(model, backend(), coding=True, harness='full', session=store)
            with pytest.raises(ContextLimitExceeded):
                await agent.context.abefore_model({
                    'messages': [HumanMessage(content='继续')],
                    'todos': [{'content': '待验证' * 100000, 'status': 'in_progress'}],
                }, None)
            assert not model.seen
    asyncio.run(run())
