import asyncio
from types import SimpleNamespace

import pytest

from cli_rb.runtime.request_budget import PersistentRequestBudget, RequestBudgetExceeded
from cli_rb.runtime.response_stream import IncompleteResponseStreamError


def test_bounded_incomplete_request_preserves_budget_on_retry_and_restart(tmp_path):
    async def run():
        limits = dict(seconds=60, requests=3, tokens=100, incomplete_token_bound=60)
        meter = PersistentRequestBudget(tmp_path / 'budget.json', **limits)
        await meter.before_http(None)
        await meter.on_llm_error(IncompleteResponseStreamError('captured truncation'))
        resumed = PersistentRequestBudget(tmp_path / 'budget.json', **limits)
        assert resumed.state['tokens'] == 60 and resumed.state['bounded_unknown_requests'] == 1
        assert not resumed.state['inflight'] and not resumed.state['unknown_usage']
        await resumed.before_http(None)
        await resumed.on_llm_error(IncompleteResponseStreamError('second truncation'))
        with pytest.raises(RequestBudgetExceeded):
            await resumed.before_http(None)
        assert resumed.state['requests'] == 2
    asyncio.run(run())


def test_real_sdk_retry_executes_only_completed_tool(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import httpx2
    from langchain.agents import create_agent
    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    import cli_rb.config as config_module
    from cli_rb.config import ModelConfig
    from cli_rb.runtime.model_recovery import ModelRequestRecovery
    from test_response_stream import payload, tool_events

    async def run():
        executed = []
        @tool
        def edit_file(path: str, old_text: str, new_text: str) -> str:
            """Edit a file."""
            executed.append('edit')
            return 'edited'
        @tool
        def read_file(path: str, start_line: int, max_lines: int) -> str:
            """Read a file."""
            executed.append('read')
            return 'read'
        events = json.loads((Path(__file__).parent / 'fixtures/tool_incomplete_response.json').read_text())
        truncated = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
        sent = []
        def serve(request):
            sent.append(request)
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'},
                                   stream=httpx2.ByteStream(truncated if len(sent) == 1 else payload(tool_events(None))))
        transport = httpx2.MockTransport(serve)
        sync = httpx2.Client(transport=transport)
        asynchronous = httpx2.AsyncClient(transport=transport)
        monkeypatch.setattr(config_module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(
            **kwargs, streaming=True, http_client=sync, http_async_client=asynchronous))
        model = ModelConfig(model='test', use_responses_api=True).build_model()
        meter = PersistentRequestBudget(tmp_path / 'budget.json', seconds=60, requests=4,
                                        tokens=100, incomplete_token_bound=60)
        meter.attach(model)
        recovery = ModelRequestRecovery(2)
        recovery.retry.initial_delay = 0
        agent = create_agent(model, [edit_file, read_file], middleware=[recovery])
        with pytest.raises(ModelCallLimitExceededError):
            await agent.ainvoke({'messages': [('user', 'Read the file')]})
        assert executed == ['read'] and len(sent) == 2
        assert meter.state['tokens'] == 72 and meter.state['bounded_unknown_requests'] == 1
        sync.close()
        await asynchronous.aclose()
    asyncio.run(run())


def returned(usage):
    return SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=usage))]])


@pytest.mark.parametrize('interruption', ['killed', 'error', 'missing_usage'])
def test_unknown_usage_survives_restart_and_prevents_new_request(tmp_path, interruption):
    async def run():
        path = tmp_path / 'budget.json'
        budget = PersistentRequestBudget(path, seconds=60, requests=3, tokens=100)
        await budget.before_http(None)
        if interruption == 'error':
            await budget.on_llm_error(RuntimeError('stream interrupted'))
        elif interruption == 'missing_usage':
            await budget.on_llm_end(returned(None))
        resumed = PersistentRequestBudget(path, seconds=60, requests=3, tokens=100)
        assert resumed.state['unknown_usage'] and resumed.state['requests'] == 1
        with pytest.raises(RequestBudgetExceeded):
            await resumed.before_http(None)
        assert resumed.state['tokens'] == 0  # Known subtotal, not actual zero consumption.
    asyncio.run(run())


def test_complete_usage_and_frozen_limits_survive_restart(tmp_path):
    async def run():
        path = tmp_path / 'budget.json'
        budget = PersistentRequestBudget(path, seconds=60, requests=3, tokens=12)
        await budget.before_http(None)
        await budget.on_llm_end(returned({'total_tokens': 12}))
        resumed = PersistentRequestBudget(path, seconds=60, requests=3, tokens=12)
        assert resumed.state['started'] == budget.state['started']
        assert resumed.state['tokens'] == 12 and not resumed.state['unknown_usage']
        with pytest.raises(RequestBudgetExceeded):
            await resumed.before_http(None)
        with pytest.raises(ValueError, match='冻结'):
            PersistentRequestBudget(path, seconds=60, requests=3, tokens=100)
    asyncio.run(run())


def test_transport_meter_counts_main_and_summary_client_copies(tmp_path, monkeypatch):
    import httpx2
    from langchain_openai import ChatOpenAI
    import cli_rb.config as config_module
    from cli_rb.config import ModelConfig
    from cli_rb.runtime.model_recovery import without_sdk_retries
    from test_response_stream import payload, tool_events

    async def run():
        sent = []
        def serve(request):
            sent.append(request)
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'},
                                   stream=httpx2.ByteStream(payload(tool_events(None))))

        transport = httpx2.MockTransport(serve)
        sync = httpx2.Client(transport=transport)
        asynchronous = httpx2.AsyncClient(transport=transport)
        monkeypatch.setattr(config_module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(
            **kwargs, streaming=True, http_client=sync, http_async_client=asynchronous))
        model = ModelConfig(model='test', use_responses_api=True).build_model()
        meter = PersistentRequestBudget(tmp_path / 'budget.json', seconds=60, requests=5, tokens=24)
        meter.attach(model)
        model = without_sdk_retries(model)
        await model.ainvoke('main')
        summary = model.model_copy(update={'max_retries': 0})
        await summary.ainvoke('summary')
        assert meter.state['requests'] == 2 and meter.state['tokens'] == 24
        assert not meter.state['inflight'] and not meter.state['unknown_usage']
        with pytest.raises(RequestBudgetExceeded):
            await model.ainvoke('must not be sent')
        assert len(sent) == 2
        sync.close()
        await asynchronous.aclose()
    asyncio.run(run())
