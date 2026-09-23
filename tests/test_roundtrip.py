from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from cli_rb.agent import build_agent, stream_turn
from cli_rb.demo import DEMO_QUESTION, DemoModel, demo_backend
from cli_rb.workspace import FixtureBackend, make_read_tool


class ScriptedModel(BaseChatModel):
    replies: list[AIMessage]
    seen: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "test-script"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies[len(self.seen) - 1])])


def request(path, call_id):
    return AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"path": path}, "id": call_id, "type": "tool_call",
    }])


def test_real_framework_roundtrip():
    messages = list(stream_turn(
        build_agent(DemoModel(), demo_backend()), [HumanMessage(content=DEMO_QUESTION)],
    ))
    assert len(messages) == 3
    assert messages[0].tool_calls[0]["name"] == "read_file"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == messages[0].tool_calls[0]["id"]
    assert "1: # CLI Rb" in messages[1].content
    assert messages[1].content in messages[2].content


def test_direct_reply_does_not_force_tool():
    model = ScriptedModel(replies=[AIMessage(content="你好")])
    messages = list(stream_turn(build_agent(model, demo_backend()), [HumanMessage(content="你好")]))
    assert len(messages) == 1
    assert messages[0].content == "你好"


def test_tool_error_returns_to_model_and_can_recover():
    model = ScriptedModel(replies=[
        request("../secret", "bad"), request("README.md", "good"), AIMessage(content="已读取"),
    ])
    messages = list(stream_turn(
        build_agent(model, demo_backend()), [HumanMessage(content=DEMO_QUESTION)],
    ))
    assert "工具错误" in model.seen[1][-1].content
    assert model.seen[1][-1].tool_call_id == "bad"
    assert "# CLI Rb" in model.seen[2][-1].content
    assert messages[-1].content == "已读取"


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "C:/secret", "a\\b", ""])
def test_paths_cannot_reach_host(path):
    result = make_read_tool(demo_backend()).invoke({"path": path})
    assert result.startswith("工具错误")


def test_paged_read_and_missing_file():
    backend = FixtureBackend({"a": "one\ntwo\nthree"})
    tool = make_read_tool(backend)
    assert tool.invoke({"path": "a", "max_lines": 2}) == (
        "1: one\n2: two\n[更多内容：从第 3 行继续读取]"
    )
    assert tool.invoke({"path": "a", "start_line": 3}) == "3: three"
    assert tool.invoke({"path": "missing"}).startswith("工具错误")
    assert tool.invoke({"path": "a", "max_lines": 201}).startswith("工具错误")
