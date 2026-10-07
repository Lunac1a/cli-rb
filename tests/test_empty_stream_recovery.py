"""Observed zero-generation failure: bounded retry without tool replay."""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from helpers import request
from cli_rb.runtime.agent import build_agent
from cli_rb.workspace.tools import FixtureBackend
from test_model_recovery import agent_for, consume


def empty():
    return ValueError('No generations found in stream.')


def test_empty_response_retries_current_request_without_tool_replay():
    agent = agent_for([request('README.md', 'read-1'), empty(), AIMessage(content='Done')], model_call_limit=3)
    result = consume(agent)
    assert result[-1].text == 'Done'
    assert sum(isinstance(m, ToolMessage) for m in result) == 1
    assert agent.model.seen[1] == agent.model.seen[2]
    assert agent.requests.calls == 3
    assert agent.requests.records[1]['usage'] is None


@pytest.mark.parametrize('limit,error,calls', [(10, ValueError, 3), (2, ModelCallLimitExceededError, 2)])
def test_persistent_empty_response_stops_inside_original_budget(limit, error, calls):
    agent = agent_for([empty() for _ in range(4)], model_call_limit=limit)
    with pytest.raises(error):
        consume(agent)
    assert len(agent.model.seen) == agent.requests.calls == calls


def test_unrelated_value_error_stops_immediately():
    agent = agent_for([ValueError('invalid parsed output'), AIMessage(content='must not run')])
    with pytest.raises(ValueError, match='invalid parsed output'):
        consume(agent)
    assert agent.requests.calls == 1


def test_actual_http_empty_sse_recovers_with_visible_usage_callbacks():
    from langchain_core.callbacks import BaseCallbackHandler
    class Calls(BaseCallbackHandler):
        starts = 0
        errors = 0
        returned = 0
        def on_chat_model_start(self, *args, **kwargs):
            self.starts += 1
        def on_llm_error(self, *args, **kwargs):
            self.errors += 1
        def on_llm_end(self, *args, **kwargs):
            self.returned += 1
    meter = Calls()
    requests = []
    def serve(req):
        requests.append(json.loads(req.content))
        if len(requests) == 1:
            body = 'data: [DONE]\n\n'
        else:
            chunk = {'id': 'chat-1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'test',
                     'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Recovered'}, 'finish_reason': None}]}
            body = f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=body)
    transport = httpx.MockTransport(serve)
    model = ChatOpenAI(model='test', api_key='test', base_url='http://test/v1', use_responses_api=False,
                       http_client=httpx.Client(transport=transport),
                       http_async_client=httpx.AsyncClient(transport=transport), callbacks=[meter])
    agent = build_agent(model, FixtureBackend({}), model_call_limit=2)
    agent.requests.retry.initial_delay = 0
    assert consume(agent)[-1].text == 'Recovered'
    assert len(requests) == meter.starts == 2
    assert meter.errors == meter.returned == 1
    assert agent.requests.records[0]['usage'] is None
