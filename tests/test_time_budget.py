import asyncio
from types import SimpleNamespace

import pytest
from helpers import ScriptedModel, StreamingModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from typer.testing import CliRunner

from cli_rb.agent import AgentRuntime, TaskTimeLimitExceeded, astream_turn, build_agent, stream_turn
from cli_rb.cli import app, display_turn
from cli_rb.demo import demo_backend


def test_default_and_hour_option(monkeypatch):
    assert build_agent(ScriptedModel(replies=[]), demo_backend()).task_timeout_seconds == 1800
    seen = []
    monkeypatch.setattr('cli_rb.cli.chat', lambda **kwargs: seen.append(kwargs))
    assert CliRunner().invoke(app, ['--task-minutes', '60']).exit_code == 0
    assert seen == [{'isolated': False, 'task_minutes': 60, 'resume': None}]
    assert CliRunner().invoke(app, ['--task-minutes', '0']).exit_code == 2


def test_deadline_before_first_event_closes_request():
    class Graph:
        closed = False

        async def astream(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
                yield
            finally:
                self.closed = True

    graph = Graph()

    async def run():
        async for _ in astream_turn(AgentRuntime(graph, None, '', task_timeout_seconds=0.1), []):
            pytest.fail('unexpected event')

    with pytest.raises(TaskTimeLimitExceeded):
        asyncio.run(run())
    assert graph.closed


def test_deadline_stops_stream_and_cleans_up_without_completing():
    model = StreamingModel(replies=[], block=True)
    cleanup, completed = [], []
    backend = SimpleNamespace(cancel_all=lambda: cleanup.append(True))
    agent = build_agent(model, demo_backend(), task_timeout_seconds=0.15)
    with pytest.raises(TaskTimeLimitExceeded):
        asyncio.run(display_turn(agent, [HumanMessage(content='你好')], backend, completed))
    assert model.closed and len(model.seen) == 1
    assert completed == [] and cleanup


def test_deadline_covers_empty_reply_recovery():
    class Model(ScriptedModel):
        closed: bool = False

        async def _agenerate(self, *args, **kwargs):
            if not self.seen:
                return self._generate(*args, **kwargs)
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True

    model = Model(replies=[AIMessage(content='')])
    with pytest.raises(TaskTimeLimitExceeded):
        list(stream_turn(build_agent(model, demo_backend(), task_timeout_seconds=0.15),
                         [HumanMessage(content='你好')]))
    assert model.closed


def test_deadline_is_not_reset_by_each_event():
    class Graph:
        count = 0

        async def astream(self, *args, **kwargs):
            for _ in range(50):
                await asyncio.sleep(0.02)
                self.count += 1
                yield 'messages', (AIMessageChunk(content='片段'), {'langgraph_node': 'model'})

    graph = Graph()
    agent = AgentRuntime(graph, None, '', task_timeout_seconds=0.12)

    async def run():
        async for _ in astream_turn(agent, []):
            pass

    with pytest.raises(TaskTimeLimitExceeded):
        asyncio.run(run())
    assert 1 <= graph.count < 50


def test_provider_timeout_is_not_misreported_as_task_deadline():
    class Graph:
        async def astream(self, *args, **kwargs):
            raise TimeoutError('provider timeout')
            yield

    async def run():
        async for _ in astream_turn(AgentRuntime(Graph(), None, ''), []):
            pass

    with pytest.raises(TimeoutError, match='provider timeout'):
        asyncio.run(run())


def test_new_turn_gets_fresh_deadline():
    model = ScriptedModel(replies=[AIMessage(content='一'), AIMessage(content='二')])
    agent = build_agent(model, demo_backend(), task_timeout_seconds=0.15)

    async def run():
        first = [e async for e in astream_turn(agent, [HumanMessage(content='一')])]
        await asyncio.sleep(0.2)
        second = [e async for e in astream_turn(agent, [HumanMessage(content='二')])]
        assert first[-1].text == '一' and second[-1].text == '二'
    asyncio.run(run())


def test_pending_approval_is_closed_without_authorizing():
    from cli_rb.approval import TerminalApproval

    async def run():
        entered, closed = asyncio.Event(), asyncio.Event()

        async def ask(action):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        approval = TerminalApproval(asyncio.get_running_loop(), ask, lambda: None)
        worker = asyncio.create_task(asyncio.to_thread(approval, 'write'))
        await asyncio.wait_for(entered.wait(), 2)
        approval.close()
        assert not await asyncio.wait_for(worker, 2)
        await asyncio.wait_for(closed.wait(), 2)
        assert not await asyncio.to_thread(approval, 'late write')
    asyncio.run(run())
