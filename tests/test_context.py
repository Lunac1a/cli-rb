import asyncio

import pytest
from helpers import ScriptedModel, request
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START

from cli_rb.agent import ContextUpdate, TaskTimeLimitExceeded, astream_turn, build_agent
from cli_rb.context import ContextLimitExceeded
from cli_rb.demo import demo_backend
from cli_rb.persistence import SessionStore

OLD = '项目代号橙湾731。没有完成验证，不能删除数据。' + '旧讨论内容。' * 1100


async def seed(agent, store):
    await agent.graph.aupdate_state(store.config, {'messages': [
        HumanMessage(content=OLD), AIMessage(content='记录了要求，验证未完成。'),
    ]}, as_node=START)
    await agent.graph.aupdate_state(store.config, None, as_node=END)
    await store.finish(agent)


def test_summary_archives_originals_and_empty_reply_recovery_uses_compact_context(tmp_path):
    async def run():
        model = ScriptedModel(replies=[AIMessage(content='项目代号橙湾731；禁止删除数据，验证未完成。'),
                                      AIMessage(content=''), AIMessage(content='橙湾731；验证未完成。')])
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver, model_call_limit=3)
                await seed(agent, store)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='代号是什么？')])]
                assert any(isinstance(e, ContextUpdate) for e in events)
                assert events[-1].text == '橙湾731；验证未完成。'
                assert len(model.seen) == 3
                for sent in model.seen[1:]:
                    assert not any(m.content == OLD for m in sent)
                    assert any(m.content == '代号是什么？' for m in sent)
                history = await store.history(agent)
                assert any(m.additional_kwargs.get('lc_source') == 'summarization' for m in history)
                archived = [s async for s in agent.graph.aget_state_history(store.config)]
                assert any(any(m.content == OLD for m in s.values.get('messages', [])) for s in archived)
        with SessionStore(tmp_path) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver)
                assert (await store.history(agent))[-1].text == '橙湾731；验证未完成。'
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['summary', 'budget', 'large_input', 'timeout'])
def test_context_stops_without_tool_execution_or_unbudgeted_calls(tmp_path, failure):
    async def run():
        class SummaryModel(ScriptedModel):
            closed: bool = False

            async def _agenerate(self, messages, *args, **kwargs):
                if failure == 'timeout':
                    self.seen.append(list(messages))
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.closed = True
                return await super()._agenerate(messages, *args, **kwargs)

        model = SummaryModel(replies=[request('README.md', 'forbidden')] if failure == 'summary'
                              else [AIMessage(content='旧记录摘要，仍有未确认结果。')])
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver,
                                    model_call_limit=1 if failure == 'budget' else 3,
                                    task_timeout_seconds=0.15 if failure == 'timeout' else 1800)
                await seed(agent, store)
                text = '新要求' if failure != 'large_input' else '过大的当前要求' * 2000
                expected = ModelCallLimitExceededError if failure == 'budget' else ContextLimitExceeded
                if failure == 'timeout':
                    expected = TaskTimeLimitExceeded
                with pytest.raises(expected):
                    async for _ in astream_turn(agent, [HumanMessage(content=text)]):
                        pass
                assert len(model.seen) == (0 if failure == 'large_input' else 1)
                assert store.data['state'] == 'blocked'
                if failure == 'timeout':
                    assert model.closed
                state = await agent.graph.aget_state(store.config)
                assert not any(isinstance(m, ToolMessage) for m in state.values['messages'])
                assert any(m.content == text for m in state.values['messages'])
    asyncio.run(run())


def test_current_request_tool_pairs_and_recovery_guard_survive_compaction(tmp_path):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            store.data.update(review_writes=True, reviewed_actions={'old-action.json': 'reviewed'})
            logs = tmp_path / 'tool-logs'
            logs.mkdir()
            (logs / 'old-action.json').write_text('{"state":"started"}')
            model = ScriptedModel(replies=[AIMessage(content='旧内容摘要。')])
            context = build_agent(model, demo_backend(), session=store).context
            user = HumanMessage(content='当前要求不能删除文件', id='current')
            call = request('README.md', 'paired')
            result = ToolMessage(content='工具输出。' * 1000, tool_call_id='paired')
            update = await context.abefore_model({'messages': [HumanMessage(content=OLD), user, call, result]}, None)
            messages = update['messages'][1:]
            assert any(m.content == user.content for m in messages)
            assert any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
            tool = next(m for m in messages if isinstance(m, ToolMessage))
            assert tool.tool_call_id == 'paired' and '截断' in tool.content
            assert result.content == '工具输出。' * 1000
            assert 'old-action.json' in context.system_message().content
            assert store.data['review_writes'] is True
    asyncio.run(run())
