import pytest
from helpers import ScriptedModel, request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cli_rb.runtime.agent import build_agent, stream_turn, has_reply, reply_status
from cli_rb.interface.demo import DEMO_QUESTION, DemoModel, demo_backend
from cli_rb.workspace.tools import FixtureBackend, make_read_tool


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
    assert tool.invoke({"path": "a", "max_lines": 400}) == "1: one\n2: two\n3: three"
    assert tool.invoke({"path": "a", "max_lines": 401}).startswith("工具参数错误")


def test_empty_reply_recovers_once_without_replaying_tools():
    backend = FixtureBackend({'README.md': 'evidence'})
    model = ScriptedModel(replies=[request('README.md', 'read'), AIMessage(content=''), AIMessage(content='已读取 evidence；没有执行测试。')])
    history = [HumanMessage(content='读文件')]
    returned = []
    for message in stream_turn(build_agent(model, backend), history):
        returned.append(message)
        history.append(message)
    assert len(model.seen) == 3
    assert sum(isinstance(m, ToolMessage) for m in returned) == 1
    assert sum(isinstance(m, ToolMessage) for m in model.seen[-1]) == 1
    assert returned[-1].response_metadata['final_reply_recovery'] == 'recovered'
    assert '没有执行测试' in returned[-1].content


@pytest.mark.parametrize('last', [AIMessage(content='  '), request('README.md', 'forbidden')])
def test_recovery_cannot_execute_tools_or_loop(last):
    model = ScriptedModel(replies=[AIMessage(content=''), last])
    returned = list(stream_turn(build_agent(model, demo_backend()), [HumanMessage(content='你好')]))
    assert len(model.seen) == 2
    assert returned[-1].content == ''
    assert not returned[-1].tool_calls
    assert returned[-1].response_metadata['final_reply_recovery'] == 'empty_or_tool_request'
    assert not any(isinstance(m, ToolMessage) for m in returned)


def test_recovery_error_preserves_existing_evidence():
    model = ScriptedModel(replies=[request('README.md', 'read'), AIMessage(content='')])
    returned = list(stream_turn(build_agent(model, demo_backend()), [HumanMessage(content='读文件')]))
    assert any(isinstance(m, ToolMessage) for m in returned)
    assert returned[-1].response_metadata['final_reply_recovery'] == 'request_failed'
    assert returned[-1].response_metadata['error_type'] == 'IndexError'


@pytest.mark.parametrize('recovery', [
    AIMessage(content='<tool_call>\n<function=read_file>\n</function>\n</tool_call>'),
    AIMessage(content='还没说完', response_metadata={'finish_reason': 'length'}),
    AIMessage(content='未修改文件；以下是格式示例：\n```xml\n<tool_call>\n<function=read_file>\n</tool_call>\n```'),
])
def test_truncated_reply_recovery_distinguishes_tool_text_from_code_examples(recovery):
    first = AIMessage(content='部分说明', response_metadata={'finish_reason': 'length'})
    model = ScriptedModel(replies=[first, recovery])
    returned = list(stream_turn(build_agent(model, demo_backend()), [HumanMessage(content='检查文件')]))
    assert not has_reply(returned[0])
    assert len(model.seen) == 2 and not any(isinstance(m, ToolMessage) for m in returned)
    assert returned[-1].response_metadata['original_reply_status'] == 'output_truncated'
    assert reply_status(returned[-1]) == reply_status(recovery)
    assert has_reply(returned[-1]) == ('```xml' in recovery.text)
