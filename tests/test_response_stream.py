"""SDK + LangChain replay of the observed done-without-delta gateway failure."""
import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from langchain_openai import ChatOpenAI

import cli_rb.config as config_module
from cli_rb.config import ModelConfig
from cli_rb.runtime.response_stream import ToolArgumentFrames, ToolArgumentStreamError
from cli_rb.runtime.response_stream import IncompleteResponseStreamError


def tool_events(delta):
    events = json.loads((Path(__file__).parent / 'fixtures/tool_done_without_delta.json').read_text())
    if delta is not None:
        done = events[1]
        events.insert(1, {'type': 'response.function_call_arguments.delta',
                         'item_id': done['item_id'], 'output_index': done['output_index'],
                         'sequence_number': 1, 'delta': delta})
    return events


def payload(events):
    response = {'id': 'resp_fixture', 'object': 'response', 'created_at': 1,
                'model': 'test', 'status': 'completed', 'output': [events[-1]['item']],
                'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
                          'input_tokens_details': {'cached_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 0}}}
    events = [{'type': 'response.created', 'response': {**response, 'output': []}},
              *events, {'type': 'response.completed', 'response': response}]
    return b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)


@pytest.mark.parametrize('delta', [None, 'partial', 'full'])
@pytest.mark.parametrize('synchronous', [False, True])
def test_configured_model_preserves_done_arguments(monkeypatch, delta, synchronous):
    events = tool_events(None)
    arguments = events[1]['arguments']
    events = tool_events(arguments[:12] if delta == 'partial' else
                         arguments if delta == 'full' else None)

    def serve(request):
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'},
                               stream=httpx2.ByteStream(payload(events)))

    transport = httpx2.MockTransport(serve)
    sync = httpx2.Client(transport=transport)
    asynchronous = httpx2.AsyncClient(transport=transport)
    monkeypatch.setattr(config_module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(
        **kwargs, streaming=True, http_client=sync, http_async_client=asynchronous))
    model = ModelConfig(model='test', use_responses_api=True).build_model()
    reply = model.invoke('read') if synchronous else asyncio.run(model.ainvoke('read'))
    assert reply.tool_calls[0]['args'] == json.loads(arguments)
    assert reply.tool_calls[0]['id'] == events[0]['item']['call_id']
    assert reply.usage_metadata['total_tokens'] == 12
    sync.close()
    asyncio.run(asynchronous.aclose())


def test_fragmented_crlf_and_regular_frames_preserve_bytes():
    events = tool_events(None)
    arguments = events[1]['arguments']
    # Fully streamed calls must pass through byte-for-byte, including other events.
    regular = payload(tool_events(arguments)).replace(b'\n', b'\r\n')
    frames = ToolArgumentFrames()
    assert b''.join(frame for byte in regular for frame in frames.feed(bytes([byte]))) == regular
    frames = ToolArgumentFrames()
    missing = payload(events).replace(b'\n', b'\r\n')
    repaired = b''.join(frame for byte in missing for frame in frames.feed(bytes([byte])))
    deltas = [json.loads(line[6:]) for line in repaired.splitlines() if line.startswith(b'data: ')]
    assert [e['delta'] for e in deltas if e['type'] == 'response.function_call_arguments.delta'] == [arguments]


def test_contradictory_done_arguments_stop_without_inventing_content():
    frames = ToolArgumentFrames()
    with pytest.raises(ToolArgumentStreamError):
        list(frames.feed(payload(tool_events('contradictory'))))


@pytest.mark.parametrize('synchronous', [False, True])
def test_captured_incomplete_stream_never_returns_tool_call(monkeypatch, synchronous):
    events = json.loads((Path(__file__).parent / 'fixtures/tool_incomplete_response.json').read_text())
    captured = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
    transport = httpx2.MockTransport(lambda request: httpx2.Response(
        200, headers={'content-type': 'text/event-stream'}, stream=httpx2.ByteStream(captured)))
    sync = httpx2.Client(transport=transport)
    asynchronous = httpx2.AsyncClient(transport=transport)
    monkeypatch.setattr(config_module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(
        **kwargs, streaming=True, http_client=sync, http_async_client=asynchronous))
    model = ModelConfig(model='test', use_responses_api=True).build_model()
    with pytest.raises(IncompleteResponseStreamError):
        model.invoke('edit') if synchronous else asyncio.run(model.ainvoke('edit'))
    sync.close()
    asyncio.run(asynchronous.aclose())


def test_done_sentinel_cannot_replace_response_completion():
    frames = ToolArgumentFrames()
    with pytest.raises(IncompleteResponseStreamError):
        list(frames.feed(b'data: [DONE]\n\n'))
