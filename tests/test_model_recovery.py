"""Fault injection: transient HTTP failures, request accounting, no tool replay."""
import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import APIStatusError

from helpers import ScriptedModel, request
from cli_rb.runtime.agent import build_agent, stream_turn, astream_turn, TaskTimeLimitExceeded
from cli_rb.runtime.model_recovery import ModelRequestRecovery
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from cli_rb.workspace.tools import FixtureBackend
from test_verification import backend as controlled_backend, call

backend = controlled_backend


def failure(status=502):
    return APIStatusError('injected HTTP failure',
                          response=httpx.Response(status, request=httpx.Request('POST', 'http://test')),
                          body=None)


class FaultModel(ScriptedModel):
    outcomes: list[Any]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        result = self.outcomes[len(self.seen) - 1]
        if isinstance(result, BaseException):
            raise result
        return ChatResult(generations=[ChatGeneration(message=result)])


def agent_for(outcomes, **kwargs):
    agent = build_agent(FaultModel(replies=[], outcomes=outcomes), FixtureBackend({'README.md': 'evidence'}), **kwargs)
    agent.requests.retry.initial_delay = 0
    return agent


def consume(agent):
    return list(stream_turn(agent, [HumanMessage(content='Inspect README then answer')]))


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
def test_transient_recovery_keeps_tool_result_and_counts_failures(status):
    agent = agent_for([request('README.md', 'read-1'), failure(status), AIMessage(content='Done')], model_call_limit=3)
    result = consume(agent)
    assert result[-1].text == 'Done'
    assert sum(isinstance(m, ToolMessage) for m in result) == 1
    assert agent.model.seen[1] == agent.model.seen[2]
    assert agent.requests.calls == 3
    assert agent.requests.records[1]['usage'] is None
    assert agent.requests.records[1]['status_code'] == status


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
def test_permanent_failure_is_not_retried(status):
    agent = agent_for([failure(status), AIMessage(content='must not run')])
    with pytest.raises(APIStatusError):
        consume(agent)
    assert len(agent.model.seen) == 1


@pytest.mark.parametrize('limit,expected_error,calls', [(10, APIStatusError, 3), (2, ModelCallLimitExceededError, 2)])
def test_retry_exhaustion_and_original_request_limit(limit, expected_error, calls):
    agent = agent_for([failure() for _ in range(4)], model_call_limit=limit)
    with pytest.raises(expected_error):
        consume(agent)
    assert len(agent.model.seen) == calls
    assert len(agent.requests.records) == calls


def test_empty_final_recovery_shares_retry_and_original_budget():
    agent = agent_for([failure(), AIMessage(content=''), failure(), AIMessage(content='Recovered')], model_call_limit=4)
    assert consume(agent)[-1].text == 'Recovered'
    assert agent.requests.calls == 4
    assert all(not isinstance(m, ToolMessage) for m in agent.model.seen[-1])
    limited = agent_for([failure(), AIMessage(content=''), failure(), AIMessage(content='must not run')], model_call_limit=3)
    with pytest.raises(ModelCallLimitExceededError):
        consume(limited)
    assert len(limited.model.seen) == 3


def test_wait_is_inside_deadline():
    agent = agent_for([failure(), AIMessage(content='must not run')], task_timeout_seconds=0.1)
    agent.requests.retry.initial_delay = 1
    with pytest.raises(TaskTimeLimitExceeded):
        consume(agent)
    assert len(agent.model.seen) == 1


def test_cancel_during_backoff_stops_without_retry():
    async def run():
        recovery = ModelRequestRecovery(5)
        reached = asyncio.Event()
        async def handler(_):
            reached.set()
            raise failure()
        task = asyncio.create_task(recovery.awrap_model_call(None, handler))
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recovery.calls == 1
    asyncio.run(run())


def test_budget_resets_for_new_turn():
    agent = agent_for([failure(), AIMessage(content='First'), AIMessage(content='Second')], model_call_limit=2)
    assert consume(agent)[-1].text == 'First'
    assert consume(agent)[-1].text == 'Second'
    assert agent.requests.calls == 1


def test_attempt_log_preserves_unknown_usage(tmp_path):
    async def run():
        recovery = ModelRequestRecovery(3, SimpleNamespace(directory=tmp_path))
        recovery.retry.initial_delay = 0
        async def handler(_):
            if recovery.calls == 1:
                raise failure()
            return AIMessage(content='ok', usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})
        await recovery.awrap_model_call(None, handler)
    asyncio.run(run())
    records = [json.loads(line) for line in (tmp_path/'model-attempts.jsonl').read_text().splitlines()]
    assert records[0]['usage'] is None
    assert records[1]['usage']['total_tokens'] == 12


def test_real_http_adapter_retries_are_visible_to_callbacks():
    requests = []
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
    def serve(req):
        requests.append(json.loads(req.content))
        if len(requests) == 1:
            return httpx.Response(502, json={'error': {'message': 'Upstream service temporarily unavailable', 'type': 'upstream_error'}})
        chunk = {'id': 'chat-1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'test',
                 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Recovered'}, 'finish_reason': None}]}
        end = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=f'data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n')
    transport = httpx.MockTransport(serve)
    model = ChatOpenAI(model='test', api_key='test', base_url='http://test/v1', max_retries=5,
                       http_client=httpx.Client(transport=transport), http_async_client=httpx.AsyncClient(transport=transport),
                       callbacks=[meter], use_responses_api=False)
    agent = build_agent(model, FixtureBackend({}), model_call_limit=2)
    agent.requests.retry.initial_delay = 0
    assert consume(agent)[-1].text == 'Recovered'
    assert len(requests) == meter.starts == 2
    assert meter.errors == meter.returned == 1
    assert agent.requests.calls == 2
    assert model.max_retries == 5  # Caller-owned model is unchanged.


def test_interrupted_stream_is_not_retried():
    from helpers import StreamingModel
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.messages import AIMessageChunk
    class BrokenStream(StreamingModel):
        async def _astream(self, messages, **kwargs):
            self.seen.append(list(messages))
            yield ChatGenerationChunk(message=AIMessageChunk(content='partial'))
            raise httpx.ReadError('stream lost')
    async def run():
        model = BrokenStream(replies=[])
        agent = build_agent(model, FixtureBackend({}))
        with pytest.raises(httpx.ReadError):
            _ = [e async for e in astream_turn(agent, [HumanMessage(content='hello')])]
        assert len(model.seen) == 1
    asyncio.run(run())


@pytest.mark.parametrize('recover', [True, False])
def test_persistent_edit_survives_recovery_or_exhaustion(backend, recover):
    from cli_rb.runtime.persistence import SessionStore
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    async def run():
        edit = call('edit_file', path='app.py', old_text='value = 1', new_text='value = 2')
        outcomes = [edit, failure()]
        outcomes += ([AIMessage(content='Changed value; verification unfinished.')] if recover
                     else [failure(), failure()])
        # Completion feedback may request a truthful partial report once.
        outcomes += [AIMessage(content='Partial: changed value, no tests run.')]
        model = FaultModel(replies=[], outcomes=outcomes)
        directory = backend.snapshot.session
        with SessionStore(directory, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(directory/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, coding=True, session=store, checkpointer=saver)
                agent.requests.retry.initial_delay = 0
                async def run_turn():
                    return [e async for e in astream_turn(agent, [HumanMessage(content='Change value to 2')])]
                if recover:
                    events = await run_turn()
                    assert isinstance(events[-1], AIMessage) and events[-1].text
                    assert store.data['state'] == 'idle'
                else:
                    with pytest.raises(APIStatusError):
                        await run_turn()
                    assert store.data['state'] == 'blocked'
                assert (backend.snapshot.workspace/'app.py').read_text() == 'value = 2\n'
                saved = await agent.graph.aget_state(store.config)
                edits = [m for m in saved.values['messages'] if isinstance(m, ToolMessage) and m.name == 'edit_file']
                assert len(edits) == 1 and not edits[0].text.startswith('工具错误')
                records = [json.loads(line) for line in (directory/'model-attempts.jsonl').read_text().splitlines()]
                assert len(records) == len(model.seen)
                assert sum(r.get('status_code') == 502 for r in records) == (1 if recover else 3)
    asyncio.run(run())
