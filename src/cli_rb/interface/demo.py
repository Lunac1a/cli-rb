"""Deterministic demonstration model. Never selected as a real-model fallback."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cli_rb.workspace.tools import FixtureBackend

DEMO_QUESTION = "请读取 README.md，告诉我项目名称和当前能力。"


def demo_backend():
    return FixtureBackend({"README.md": "# CLI Rb\n当前能力：演示模型与只读工具的往返。\n"})


class DemoModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "explicit-scripted-demo"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            answer = AIMessage(content=f"演示模型收到的工具结果：\n{messages[-1].content}")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "id": "demo-read-1",
                    "type": "tool_call",
                }],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])
