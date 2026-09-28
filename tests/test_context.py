import asyncio
import json

import pytest
from helpers import ScriptedModel, request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START

from cli_rb.runtime.agent import ContextUpdate, astream_turn, build_agent
from cli_rb.runtime.context import ContextLimitExceeded, ContextPolicy, estimate
from cli_rb.interface.demo import demo_backend
from cli_rb.runtime.persistence import SessionStore

OLD = '旧讨论内容。' * 4200


@pytest.fixture(autouse=True)
def bounded_context_fixture(monkeypatch):
    # Exercise pressure branches with the original small synthetic fixtures.
    # These are boundary tests, not the production working-window default.
    monkeypatch.setenv('CLI_RB_CONTEXT_WINDOW', '32768')


async def seed(agent, store):
    await agent.graph.aupdate_state(store.config, {'messages': [
        HumanMessage(content='项目代号橙湾731，不能删除数据。'), AIMessage(content=OLD),
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
def test_optional_summary_fallback_and_hard_limits(tmp_path, failure):
    async def run():
        class SlowModel(ScriptedModel):
            async def _agenerate(self, messages, *args, **kwargs):
                if len(self.seen) == 0 and failure == 'timeout':
                    self.seen.append(list(messages))
                    await asyncio.Event().wait()
                return await super()._agenerate(messages, *args, **kwargs)

        replies = ([AIMessage(content='直接回答')] if failure == 'budget' else
                   [request('README.md', 'must-not-execute'), AIMessage(content='继续回答')])
        model = SlowModel(replies=replies)
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver,
                                    model_call_limit=1 if failure == 'budget' else 3,
                                    context_policy=ContextPolicy(window=32768, summary_seconds=0.02))
                await seed(agent, store)
                text = '过大' * 20000 if failure == 'large_input' else '继续'
                if failure == 'large_input':
                    with pytest.raises(ContextLimitExceeded):
                        _ = [e async for e in astream_turn(agent, [HumanMessage(content=text)])]
                    assert not model.seen
                else:
                    events = [e async for e in astream_turn(agent, [HumanMessage(content=text)])]
                    assert events[-1].text in {'直接回答', '继续回答'}
                    assert len(model.seen) == (1 if failure == 'budget' else 2)
                    state = await agent.graph.aget_state(store.config)
                    assert not any(isinstance(m, ToolMessage) for m in state.values['messages'])
                    assert any(m.content == OLD for m in state.values['messages'])
                    assert state.values['thread_model_call_count'] == len(model.seen)
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
            update = await context.abefore_model({'messages': [HumanMessage(content='早期约束'), AIMessage(content=OLD), user, call, result]}, None)
            messages = update['messages'][1:]
            assert any(m.content == user.content for m in messages)
            assert any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
            tool = next(m for m in messages if isinstance(m, ToolMessage))
            assert tool.tool_call_id == 'paired' and tool.content == result.content
            assert result.content == '工具输出。' * 1000
            assert 'old-action.json' in context.system_message().content
            assert store.data['review_writes'] is True
    asyncio.run(run())


def test_full_code_page_reaches_next_model_call_without_early_summary(tmp_path):
    from cli_rb.workspace.tools import FixtureBackend

    async def run():
        code = '\n'.join(f'key_{i} = "value_{i}"  # relevant implementation' for i in range(100))
        backend = FixtureBackend({'README.md': code})
        expected = backend.read_file('README.md', 1, 200)
        call = request('README.md', 'read-full')
        call.tool_calls[0]['args']['max_lines'] = 200
        model = ScriptedModel(replies=[call, AIMessage(content='Read complete.')])
        with SessionStore(tmp_path, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, session=store, checkpointer=saver)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='Inspect the code.')])]
                assert len(model.seen) == 2
                assert not any(isinstance(e, ContextUpdate) for e in events)
                tool = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
                assert tool.content == expected and 'key_50' in tool.content
        records = [json.loads(line) for line in (tmp_path/'context-calls.jsonl').read_text().splitlines()]
        assert [r['kind'] for r in records] == ['model', 'model']
    asyncio.run(run())


def test_estimation_keeps_unicode_conservative_and_counts_tool_arguments():
    assert estimate([HumanMessage(content='中' * 2000)]) >= 2000
    assert 1000 <= estimate([HumanMessage(content='a' * 2000)]) < 1100
    call = request('README.md', 'large-args')
    call.tool_calls[0]['args']['path'] = 'x' * 4000
    assert estimate([call]) > 2000


def test_oversized_latest_observation_fails_without_erasing_or_summarizing_it(tmp_path):
    async def run():
        model = ScriptedModel(replies=[])
        with SessionStore(tmp_path, create=True) as store:
            context = build_agent(model, demo_backend(), session=store).context
            result = ToolMessage(content='x' * 60000, tool_call_id='oversize')
            with pytest.raises(ContextLimitExceeded):
                await context.abefore_model({'messages': [
                    HumanMessage(content=OLD), HumanMessage(content='Current task'),
                    request('README.md', 'oversize'), result,
                ]}, None)
            assert not model.seen
            assert result.content == 'x' * 60000
    asyncio.run(run())


@pytest.mark.parametrize('reply,reason', [
    (AIMessage(content='', response_metadata={'finish_reason': 'length'}), 'output_limit'),
    (AIMessage(content='Partial summary', response_metadata={'status': 'incomplete'}), 'incomplete_response'),
    (AIMessage(content=''), 'empty_text'),
])
def test_summary_failure_records_specific_reason_without_accepting_it(tmp_path, reply, reason):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            model = ScriptedModel(replies=[reply])
            context = build_agent(model, demo_backend(), session=store).context
            with pytest.raises(ContextLimitExceeded, match=reason):
                await context._acreate_summary([HumanMessage(content='Older history')])
            record = json.loads((tmp_path/'context-calls.jsonl').read_text())
            assert record['outcome'] == reason
            assert record['seconds'] >= 0
            assert 'content' not in record
    asyncio.run(run())


@pytest.mark.parametrize('ratio,expected', [(0.79, 0), (0.84, 0), (0.86, 1)])
def test_pressure_boundary_and_failed_attempt_latch(tmp_path, ratio, expected):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            model = ScriptedModel(replies=[AIMessage(content='')])
            context = build_agent(model, demo_backend(), session=store).context
            padding = int(context.policy.input_limit * ratio - context.total_size([
                AIMessage(content=''), HumanMessage(content='current')]))
            messages = [AIMessage(content='a' * (padding * 2)), HumanMessage(content='current')]
            state = {'messages': messages}
            update = await context.abefore_model(state, None)
            assert len(model.seen) == expected
            assert 'messages' not in update
            # Recreate middleware to exercise durable failure latch, without graph update.
            rebuilt = build_agent(model, demo_backend(), session=store).context
            await rebuilt.abefore_model(state, None)
            assert len(model.seen) == expected
    asyncio.run(run())


def test_failed_summary_at_hard_limit_preserves_attempt_and_originals(tmp_path):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            model = ScriptedModel(replies=[AIMessage(content='')])
            context = build_agent(model, demo_backend(), session=store).context
            messages = [AIMessage(content='a' * 54000), HumanMessage(content='keep')]
            with pytest.raises(ContextLimitExceeded):
                await context.abefore_model({'messages': messages}, None)
            assert messages[0].content == 'a' * 54000
            assert store.data['context_maintenance']['attempts'] == 1
            with pytest.raises(ContextLimitExceeded):
                await context.abefore_model({'messages': messages}, None)
            assert len(model.seen) <= 1
    asyncio.run(run())


def test_reject_summary_that_increases_context(tmp_path):
    async def run():
        with SessionStore(tmp_path, create=True) as store:
            model = ScriptedModel(replies=[AIMessage(content='中' * 26000)])
            context = build_agent(model, demo_backend(), session=store).context
            messages = [AIMessage(content=OLD), HumanMessage(content='keep')]
            update = await context.abefore_model({'messages': messages}, None)
            assert 'messages' not in update
            assert update['context_summary_attempts'] == 1
    asyncio.run(run())


def test_policy_capacity_accounts_for_output_and_margin():
    assert ContextPolicy(window=32768).input_limit == 26624
    assert ContextPolicy(window=32768, output_reserve=8192).input_limit == 22528
    with pytest.raises(ValueError):
        ContextPolicy(window=4096)


def test_default_working_window_and_explicit_override(monkeypatch):
    monkeypatch.delenv('CLI_RB_ENV_FILE', raising=False)
    monkeypatch.delenv('CLI_RB_CONTEXT_WINDOW')
    assert ContextPolicy().window == ContextPolicy.from_env().window == 200000
    assert ContextPolicy.from_env().input_limit == 193856
    monkeypatch.setenv('CLI_RB_CONTEXT_WINDOW', '32768')
    assert ContextPolicy.from_env().input_limit == 26624


def test_profile_window_reaches_model_context_budget(tmp_path, monkeypatch):
    from cli_rb.config import ModelConfig
    from cli_rb.runtime.context import ContextMiddleware

    profile = tmp_path / 'model.env'
    profile.write_text('CLI_RB_MODEL=test\nCLI_RB_CONTEXT_WINDOW=200000\n'
                       'CLI_RB_MAX_OUTPUT_TOKENS=32768\n')
    for name in ('CLI_RB_CONTEXT_WINDOW', 'CLI_RB_MAX_OUTPUT_TOKENS', 'CLI_RB_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CLI_RB_ENV_FILE', str(profile))
    model = ModelConfig.from_env().build_model()
    context = ContextMiddleware(model, 'test', [], 50, None)
    assert context.policy.window == 200000
    assert context.policy.input_limit == 165184
    monkeypatch.setenv('CLI_RB_CONTEXT_WINDOW', '64000')
    assert ContextMiddleware(model, 'test', [], 50, None).policy.input_limit == 29184


@pytest.mark.parametrize("output_cap,reserve", [(2048, 4096), (8192, 8192), (32768, 32768)])
def test_context_reserves_actual_model_output(output_cap, reserve):
    from langchain_openai import ChatOpenAI
    from cli_rb.runtime.context import ContextMiddleware
    model = ChatOpenAI(model="test", api_key="test-only", max_tokens=output_cap)
    middleware = ContextMiddleware(model, "test", [], 50, None,
                                   policy=ContextPolicy(window=200000))
    assert middleware.policy.output_reserve == reserve
    assert middleware.policy.input_limit + output_cap + 2048 <= 200000
