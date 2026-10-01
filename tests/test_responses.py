import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from cli_rb.runtime.agent import build_agent, has_reply, reply_status, stream_turn
from cli_rb.config import ModelConfig
from cli_rb.workspace.tools import FixtureBackend


@pytest.mark.parametrize('transient_failure', [False, True])
def test_responses_streaming_tool_roundtrip_preserves_reasoning(monkeypatch, transient_failure):
    requests = []

    def serve(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == '/v1/responses'
        assert payload['stream'] is True
        assert payload['store'] is False
        assert payload['reasoning'] == {'effort': 'medium'}
        assert payload['max_output_tokens'] == 32768
        assert 'temperature' not in payload
        if transient_failure and len(requests) == 2:
            return httpx.Response(502, json={'error': {'message': 'Upstream service temporarily unavailable', 'type': 'upstream_error'}})
        first = len(requests) == 1
        reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [],
                     'encrypted_content': 'opaque-test-reasoning'}
        call = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1',
                'name': 'read_file', 'arguments': '{"path":"README.md"}', 'status': 'completed'}
        message = {'type': 'message', 'id': 'msg_2', 'role': 'assistant', 'status': 'completed',
                   'content': [{'type': 'output_text', 'text': 'Read evidence.', 'annotations': []}]}
        output = [reasoning, call] if first else [message]
        response = {'id': f'resp_{len(requests)}', 'object': 'response', 'created_at': 1,
                    'model': 'test-reasoning', 'status': 'completed', 'output': output,
                    'parallel_tool_calls': True, 'tool_choice': 'auto', 'tools': [],
                    'usage': {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30,
                              'input_tokens_details': {'cached_tokens': 5},
                              'output_tokens_details': {'reasoning_tokens': 3}}}
        events = [{'type': 'response.created', 'response': {**response, 'output': [], 'status': 'in_progress'}}]
        for index, item in enumerate(output):
            events.append({'type': 'response.output_item.added', 'output_index': index, 'item': item})
            if item['type'] == 'message':
                events.append({'type': 'response.output_text.delta', 'output_index': index,
                               'content_index': 0, 'item_id': item['id'], 'delta': 'Read evidence.'})
            events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
        events.append({'type': 'response.completed', 'response': response})
        sse = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=sse)

    transport = httpx.MockTransport(serve)
    monkeypatch.setattr('cli_rb.config.ChatOpenAI', lambda **kwargs: ChatOpenAI(
        **kwargs, http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport)))
    monkeypatch.delenv('CLI_RB_API_KEY', raising=False)
    monkeypatch.delenv('CLI_RB_ENV_FILE', raising=False)
    monkeypatch.setenv('CLI_RB_USE_RESPONSES_API', '1')
    monkeypatch.setenv('CLI_RB_REASONING_EFFORT', 'medium')
    monkeypatch.setenv('CLI_RB_MODEL', 'test-reasoning')
    model = ModelConfig.from_env().build_model()
    result = list(stream_turn(build_agent(model, FixtureBackend({'README.md': 'evidence'})),
                              [HumanMessage(content='Read README.md then answer.')]))
    assert len(requests) == (3 if transient_failure else 2)
    if transient_failure:
        assert requests[1] == requests[2]
        assert sum(isinstance(item, ToolMessage) for item in result) == 1
    assert any(isinstance(item, ToolMessage) and item.tool_call_id == 'call_1' for item in result)
    assert result[-1].text == 'Read evidence.'
    assert result[-1].usage_metadata['output_token_details']['reasoning'] == 3
    history = requests[1]['input']
    assert any(item.get('encrypted_content') == 'opaque-test-reasoning' for item in history)
    assert any(item.get('type') == 'function_call_output' and item['call_id'] == 'call_1'
               and 'evidence' in item['output'] for item in history)
    assert 'previous_response_id' not in requests[1]


@pytest.mark.parametrize('status', ['incomplete', 'failed', 'cancelled'])
def test_responses_unfinished_text_is_not_a_complete_reply(status):
    message = AIMessage(content='Partial answer', response_metadata={'status': status})
    assert reply_status(message) == 'incomplete_reply'
    assert not has_reply(message)
