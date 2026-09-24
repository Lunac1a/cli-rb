import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from helpers import ScriptedModel, StreamingModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from cli_rb.agent import TextDelta, astream_turn, build_agent
from cli_rb.cli import display_turn
from cli_rb.demo import demo_backend


def test_deltas_arrive_before_completion_and_do_not_enter_history():
    async def check():
        model = StreamingModel(replies=[], block=True, release=asyncio.Event())
        history = [HumanMessage(content="你好")]
        events = astream_turn(build_agent(model, demo_backend()), history)
        first = await asyncio.wait_for(anext(events), 3)
        assert first == TextDelta("第一段")
        assert not model.closed
        assert len(history) == 1
        model.release.set()
        rest = [event async for event in events]
        assert rest[0] == TextDelta("第二段")
        assert isinstance(rest[-1], AIMessage)
        assert rest[-1].text == "第一段第二段"
        assert model.closed
    asyncio.run(check())


def test_fragmented_tool_arguments_only_execute_as_complete_call():
    async def check():
        model = StreamingModel(replies=[], tool_roundtrip=True)
        history = [HumanMessage(content="读 README")]
        events = []
        async for event in astream_turn(build_agent(model, demo_backend()), history):
            events.append(event)
            if not isinstance(event, TextDelta):
                history.append(event)
        assert events[0].tool_calls[0]["args"] == {"path": "README.md"}
        assert isinstance(events[1], ToolMessage)
        assert "# CLI Rb" in events[1].text
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["第一段", "第二段"]
        assert len(model.seen) == 2
        assert sum(isinstance(m, ToolMessage) for m in model.seen[-1]) == 1
        assert len(history) == 4
    asyncio.run(check())


def test_render_stream_once_and_hide_reasoning(monkeypatch):
    output = StringIO()
    monkeypatch.setattr("cli_rb.cli.console", Console(file=output, width=120))
    model = StreamingModel(replies=[])
    history, completed = [HumanMessage(content="你好")], []
    asyncio.run(display_turn(build_agent(model, demo_backend()), history,
                             SimpleNamespace(cancel_all=lambda: None), completed))
    assert output.getvalue().count("第一段第二段") == 1
    assert "PRIVATE_REASONING" not in output.getvalue()
    assert len(completed) == 1 and history[-1] is completed[-1]


def test_cancel_closes_model_stream_and_does_not_complete_or_recover():
    async def check():
        model = StreamingModel(replies=[], block=True, reached=asyncio.Event())
        cancelled = []
        history, completed = [HumanMessage(content="你好")], []
        task = asyncio.create_task(display_turn(build_agent(model, demo_backend()), history,
                                   SimpleNamespace(cancel_all=lambda: cancelled.append(True)), completed))
        await asyncio.wait_for(model.reached.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert model.closed and cancelled == [True]
        assert completed == [] and len(history) == 1
        assert len(model.seen) == 1
    asyncio.run(check())


def test_cancellation_during_empty_reply_recovery_is_not_swallowed():
    class RecoveryModel(ScriptedModel):
        reached: object
        closed: bool = False

        async def _agenerate(self, *args, **kwargs):
            if not self.seen:
                return self._generate(*args, **kwargs)
            self.reached.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True

    async def check():
        model = RecoveryModel(replies=[AIMessage(content="")], reached=asyncio.Event())
        history, completed, cancelled = [HumanMessage(content="你好")], [], []
        task = asyncio.create_task(display_turn(build_agent(model, demo_backend()), history,
                                   SimpleNamespace(cancel_all=lambda: cancelled.append(True)), completed))
        await asyncio.wait_for(model.reached.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert model.closed and cancelled == [True]
        assert len(completed) == 1
        assert "final_reply_recovery" not in completed[0].response_metadata
    asyncio.run(check())
