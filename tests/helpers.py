"""Small deterministic models shared by agent tests; no real model calls."""
import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


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



class StreamingModel(ScriptedModel):
    closed: bool = False
    block: bool = False
    tool_roundtrip: bool = False
    reached: object = None
    release: object = None

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        try:
            if self.tool_roundtrip and not isinstance(messages[-1], ToolMessage):
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": "read_file", "args": '{"path":', "id": "read-1", "index": 0},
                ]))
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": None, "args": '"README.md"}', "id": None, "index": 0},
                ]))
                return
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "PRIVATE_REASONING"},
            ))
            yield ChatGenerationChunk(message=AIMessageChunk(content="第一段"))
            if self.reached:
                self.reached.set()
            if self.block:
                await (self.release or asyncio.Event()).wait()
            yield ChatGenerationChunk(message=AIMessageChunk(content="第二段"))
        finally:
            self.closed = True
