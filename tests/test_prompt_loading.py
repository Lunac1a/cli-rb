from types import SimpleNamespace

import pytest
from helpers import ScriptedModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cli_rb.runtime.agent import build_agent, stream_turn


@pytest.mark.parametrize('mode', ['direct', 'isolated'])
def test_coding_prompt_reaches_model_and_empty_reply_recovery_with_actual_limits(mode):
    backend = SimpleNamespace(snapshot=SimpleNamespace(mode=mode))
    model = ScriptedModel(replies=[AIMessage(content=''), AIMessage(content='未执行测试。')])
    agent = build_agent(model, backend, coding=True, model_call_limit=7,
                        tool_call_limit=13, task_timeout_seconds=90)
    replies = list(stream_turn(agent, [HumanMessage(content='说明当前执行范围')]))
    assert replies[-1].content == '未执行测试。'
    assert len(model.seen) == 2
    for messages in model.seen:
        prompt = next(m.text for m in messages if isinstance(m, SystemMessage))
        # Framework planning appends a separate block; base instructions remain intact.
        assert prompt.startswith(agent.prompt)
        assert '{{' not in prompt
        assert '7次模型调用、13次工具调用、90秒' in prompt
        if mode == 'direct':
            assert '当前为直接模式' in prompt
            assert 'run_command 对工作区只读' in prompt
            assert '原仓库不挂载' not in prompt
        else:
            assert '当前为隔离模式' in prompt
            assert '原仓库不挂载' in prompt
            assert '普通编辑立即生效' not in prompt
