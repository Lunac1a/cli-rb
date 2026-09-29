"""Exercise native compaction through real HTTP serialization and graph checkpoints."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END

from cli_rb.interface.demo import demo_backend
from cli_rb.runtime.agent import build_agent, astream_turn, ContextUpdate
from cli_rb.runtime.context import ContextPolicy, estimate
from cli_rb.runtime.persistence import SessionStore


OLD = 'Earlier assistant working history. ' * 1400
COMPACT = {'type': 'compaction', 'id': 'cmp_test', 'encrypted_content': 'opaque' * 2000}


@pytest.fixture
def session_dir():
    # Ordinary directories avoid Windows ACL problems with pytest's private temp dirs.
    path = Path('.cli-rb/native-compaction-tests') / uuid4().hex
    path.mkdir(parents=True)
    return path


def sse(output, *, status='completed'):
    response = {'id': 'resp_' + uuid4().hex, 'object': 'response', 'created_at': 1,
                'model': 'test', 'status': status, 'output': output,
                'usage': {'input_tokens': 20000, 'output_tokens': 80, 'total_tokens': 20080,
                          'input_tokens_details': {'cached_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 0}}}
    events = [{'type': 'response.created', 'response': {**response, 'output': [], 'status': 'in_progress'}}]
    for index, item in enumerate(output):
        events.append({'type': 'response.output_item.added', 'output_index': index, 'item': item})
        if item['type'] == 'message':
            events.append({'type': 'response.output_text.delta', 'output_index': index,
                           'content_index': 0, 'item_id': item['id'],
                           'delta': item['content'][0]['text']})
        events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    events.append({'type': 'response.' + status, 'response': response})
    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=''.join(
        f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events))


def model_for(serve):
    return ChatOpenAI(model='test', api_key='test', base_url='http://test/v1',
                      max_tokens=4096, max_retries=0, use_responses_api=True,
                      output_version='responses/v1', store=False,
                      http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)))


def test_native_compaction_tool_roundtrip_and_checkpoint_resume(session_dir):
    requests = []

    def serve(request):
        data = json.loads(request.content)
        requests.append(data)
        assert request.url.path == '/v1/responses'
        if len(requests) == 1:
            assert data['input'][-1] == {'type': 'compaction_trigger'}
            assert OLD in json.dumps(data)
            assert data['tools'] and data['instructions']
            return sse([COMPACT])
        assert any(i.get('encrypted_content') == COMPACT['encrypted_content'] for i in data['input'])
        assert not any(i.get('type') == 'compaction_trigger' for i in data['input'])
        assert OLD not in json.dumps(data)
        assert 'Keep all data.' in json.dumps(data)
        if len(requests) == 2:
            return sse([{'type': 'function_call', 'id': 'fc_test', 'call_id': 'call_test',
                         'name': 'read_file', 'arguments': '{"path":"README.md"}', 'status': 'completed'}])
        assert any(i.get('type') == 'function_call_output' and i['call_id'] == 'call_test'
                   for i in data['input'])
        return sse([{'type': 'message', 'id': 'msg_test', 'role': 'assistant', 'status': 'completed',
                     'content': [{'type': 'output_text', 'text': 'Read complete.', 'annotations': []}]}])

    async def run():
        model = model_for(serve)
        policy = ContextPolicy(window=32768, strategy='codex_v2')
        with SessionStore(session_dir, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(session_dir/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver,
                                    context_policy=policy, model_call_limit=4)
                await agent.graph.aupdate_state(store.config, {'messages': [
                    HumanMessage(content='Keep all data.'), AIMessage(content=OLD)]}, as_node=START)
                await agent.graph.aupdate_state(store.config, None, as_node=END)
                await store.finish(agent)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='Read README.md.')])]
                assert any(isinstance(e, ContextUpdate) for e in events)
                assert events[-1].text == 'Read complete.'
                state = await agent.graph.aget_state(store.config)
                assert state.values['thread_model_call_count'] == 3
                compact = next(m for m in state.values['messages'] if m.additional_kwargs.get('lc_source') == 'codex_v2')
                assert estimate([compact]) < 150  # ciphertext is not counted as plaintext
                archived = [s async for s in agent.graph.aget_state_history(store.config)]
                assert any(any(m.content == OLD for m in s.values.get('messages', [])) for s in archived)
                await store.finish(agent)
        with SessionStore(session_dir) as store:
            async with AsyncSqliteSaver.from_conn_string(str(session_dir/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver, context_policy=policy)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='Continue from saved state.')])]
                assert events[-1].text == 'Read complete.'
        assert len(requests) == 4
        await model.root_async_client.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['missing_item', 'incomplete', 'timeout'])
def test_native_failure_keeps_original_and_does_not_retry(session_dir, failure):
    calls = []

    def serve(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('test timeout', request=request)
        return sse([] if failure == 'missing_item' else [COMPACT],
                   status='incomplete' if failure == 'incomplete' else 'completed')

    async def run():
        model = model_for(serve)
        with SessionStore(session_dir, create=True) as store:
            context = build_agent(model, demo_backend(), session=store,
                                  context_policy=ContextPolicy(window=32768, strategy='codex_v2')).context
            messages = [HumanMessage(content='Keep data.', id='user'), AIMessage(content=OLD)]
            update = await context.abefore_model({'messages': messages}, None)
            assert 'messages' not in update and update['run_model_call_count'] == 1
            assert messages[-1].content == OLD
            await context.abefore_model({'messages': messages, **update}, None)
            assert len(calls) == 1
        await model.root_async_client.close()
    asyncio.run(run())


def test_native_respects_call_budget(session_dir):
    def serve(request):
        pytest.fail('No maintenance call allowed with only one call remaining')

    async def run():
        model = model_for(serve)
        with SessionStore(session_dir, create=True) as store:
            context = build_agent(model, demo_backend(), session=store, model_call_limit=1,
                                  context_policy=ContextPolicy(window=32768, strategy='codex_v2')).context
            update = await context.abefore_model({'messages': [
                HumanMessage(content='Keep data.'), AIMessage(content=OLD)]}, None)
            assert 'messages' not in update and update['context_summary_attempts'] == 0
        await model.root_async_client.close()
    asyncio.run(run())


def test_native_keeps_latest_tool_group_and_replays_previous_compaction(session_dir):
    def serve(request):
        payload = json.loads(request.content)
        assert any(i.get('type') == 'compaction' for i in payload['input'])
        assert any(i.get('type') == 'function_call_output' for i in payload['input'])
        return sse([{**COMPACT, 'id': 'cmp_new'}])

    async def run():
        model = model_for(serve)
        with SessionStore(session_dir, create=True) as store:
            context = build_agent(model, demo_backend(), session=store,
                                  context_policy=ContextPolicy(window=32768, strategy='codex_v2')).context
            prior = AIMessage(content=[COMPACT], additional_kwargs={
                'lc_source': 'codex_v2', 'context_compaction_tokens': 96})
            call = AIMessage(content='', tool_calls=[{
                'name': 'read_file', 'args': {'path': 'README.md'}, 'id': 'call_latest'}])
            result = ToolMessage(content='Current observation, not yet consumed.', tool_call_id='call_latest')
            update = await context.abefore_model({'messages': [
                HumanMessage(content='Keep data.'), prior, AIMessage(content=OLD), call, result]}, None)
            assert update['messages'][-2:] == [call, result]
            assert prior not in update['messages']
            assert any(m.additional_kwargs.get('lc_source') == 'codex_v2' for m in update['messages'])
        await model.root_async_client.close()
    asyncio.run(run())
