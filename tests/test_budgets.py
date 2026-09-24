import asyncio
from types import SimpleNamespace

import pytest
from helpers import ScriptedModel, request
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cli_rb.agent import build_agent, stream_turn
from cli_rb.cli import chat_loop
from cli_rb.demo import demo_backend
from cli_rb.workspace import FixtureBackend


def test_loop_stops_before_next_model_call_and_does_not_recover():
    limit = 50
    model = ScriptedModel(replies=[request('README.md', f'read-{i}') for i in range(limit + 1)])
    received = []

    async def run():
        from cli_rb.agent import astream_turn
        async for event in astream_turn(build_agent(model, demo_backend(), model_call_limit=limit),
                                       [HumanMessage(content='读取')]):
            received.append(event)

    with pytest.raises(ModelCallLimitExceededError) as stopped:
        asyncio.run(run())
    assert stopped.value.run_count == limit
    assert len(model.seen) == limit
    assert sum(isinstance(m, ToolMessage) for m in received) == limit


def test_empty_reply_cannot_bypass_budget_via_recovery():
    model = ScriptedModel(replies=[AIMessage(content=''), AIMessage(content='不得调用')])
    with pytest.raises(ModelCallLimitExceededError):
        list(stream_turn(build_agent(model, demo_backend(), model_call_limit=1),
                         [HumanMessage(content='你好')]))
    assert len(model.seen) == 1


def test_recovery_can_use_last_available_call():
    model = ScriptedModel(replies=[AIMessage(content=''), AIMessage(content='补全')])
    returned = list(stream_turn(build_agent(model, demo_backend(), model_call_limit=2),
                               [HumanMessage(content='你好')]))
    assert returned[-1].response_metadata['final_reply_recovery'] == 'recovered'
    assert len(model.seen) == 2


def test_chat_reports_limit_and_cleans_up_without_next_prompt(capsys):
    model = ScriptedModel(replies=[AIMessage(content='')])
    prompts, cleanup = [], []
    session = SimpleNamespace(prompt=lambda _: prompts.append(True) or '你好')
    backend = SimpleNamespace(cancel_all=lambda: cleanup.append(True), command_evidence=[])
    with asyncio.Runner() as runner:
        chat_loop(session, build_agent(model, demo_backend(), model_call_limit=1), backend, [], runner)
    assert prompts == [True] and cleanup == [True]
    assert '额度已用完（1/1）' in capsys.readouterr().out
    assert len(model.seen) == 1


class CountingBackend(FixtureBackend):
    def __init__(self):
        super().__init__({'README.md': 'evidence'})
        self.calls = []
        self.command_evidence = []
        self.cancelled = False

    def read_file(self, path, start_line=1, max_lines=80):
        self.calls.append(path)
        return super().read_file(path, start_line, max_lines)

    def cancel_all(self):
        self.cancelled = True


def batch(count):
    return AIMessage(content='', tool_calls=[
        request('README.md', f'read-{i}').tool_calls[0] for i in range(count)
    ])


def test_exact_batch_allowance_then_normal_answer():
    limit = 100
    backend = CountingBackend()
    model = ScriptedModel(replies=[batch(limit), AIMessage(content='已读取')])
    returned = list(stream_turn(build_agent(model, backend, tool_call_limit=limit, model_call_limit=2),
                               [HumanMessage(content='读文件')]))
    assert len(backend.calls) == limit
    assert returned[-1].text == '已读取'


def test_oversized_batch_is_entirely_blocked_without_recovery():
    prior = 1
    backend = CountingBackend()
    model = ScriptedModel(replies=[request('README.md', 'prior'), batch(2)])
    with pytest.raises(ToolCallLimitExceededError) as stopped:
        list(stream_turn(build_agent(model, backend, tool_call_limit=prior + 1),
                         [HumanMessage(content='读取')]))
    assert stopped.value.run_count == prior + 2  # Requested, not executed.
    assert len(backend.calls) == prior
    assert len(model.seen) == prior + 1


def test_failed_tool_still_consumes_allowance():
    backend = CountingBackend()
    model = ScriptedModel(replies=[request('missing', 'bad'), request('README.md', 'again')])
    with pytest.raises(ToolCallLimitExceededError):
        list(stream_turn(build_agent(model, backend, tool_call_limit=1),
                         [HumanMessage(content='读取')]))
    assert backend.calls == ['missing']
    assert '工具错误' in model.seen[-1][-1].text


def test_new_user_request_resets_model_and_tool_allowances():
    backend = CountingBackend()
    model = ScriptedModel(replies=[batch(1), AIMessage(content='第一轮'),
                                  batch(1), AIMessage(content='第二轮')])
    agent = build_agent(model, backend, tool_call_limit=1, model_call_limit=2)
    history = [HumanMessage(content='读文件')]
    history.extend(stream_turn(agent, history))
    history.append(HumanMessage(content='再读'))
    assert list(stream_turn(agent, history))[-1].text == '第二轮'
    assert len(backend.calls) == 2


def test_cli_reports_requested_count_not_executed_count(capsys):
    backend = CountingBackend()
    model = ScriptedModel(replies=[batch(2)])
    with asyncio.Runner() as runner:
        chat_loop(SimpleNamespace(prompt=lambda _: '读取'),
                  build_agent(model, backend, tool_call_limit=1), backend, [], runner)
    text = capsys.readouterr().out
    assert '累计请求 2' in text and '本批工具全部未执行' in text
    assert backend.cancelled and not backend.calls
