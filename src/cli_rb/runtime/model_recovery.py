"""Retry only model requests, with each attempt charged to the existing allowance."""
import json
import time

from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from openai import APIStatusError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def transient_model_error(error):
    # HTTP rejection before a response stream. Do not retry interrupted streams
    # or ambiguous transport timeouts after partial output has been displayed.
    return isinstance(error, APIStatusError) and error.status_code in {408, 429, 500, 502, 503, 504}


def without_sdk_retries(model):
    if not isinstance(model, ChatOpenAI):
        return model
    sync = model.root_client.with_options(max_retries=0)
    asynchronous = model.root_async_client.with_options(max_retries=0)
    return model.model_copy(update={'max_retries': 0, 'root_client': sync,
                                   'client': sync.chat.completions,
                                   'root_async_client': asynchronous,
                                   'async_client': asynchronous.chat.completions})


class ModelRequestRecovery(ModelCallLimitMiddleware):
    """One runtime serves one active turn; tools are outside this retry boundary."""

    def __init__(self, limit, session=None, *, reserve_final=False):
        super().__init__(run_limit=limit, exit_behavior='error')
        self.calls = 0
        self.records = []
        self.session = session
        self.reserve_final = reserve_final
        self.retry = ModelRetryMiddleware(max_retries=2, retry_on=transient_model_error,
                                         on_failure='error', initial_delay=1, max_delay=2,
                                         jitter=False)

    def before_agent(self, state, runtime):
        self.calls = 0
        self.records = []

    async def abefore_agent(self, state, runtime):
        self.before_agent(state, runtime)

    def after_model(self, state, runtime):
        return {'run_model_call_count': self.calls,
                'thread_model_call_count': state.get('thread_model_call_count', 0)
                + self.calls - state.get('run_model_call_count', 0)}

    async def awrap_model_call(self, request, handler):
        # Context maintenance increments graph counts outside this wrapper.
        # Absorb it before charging any main request or retry.
        if request is not None:
            self.calls = max(self.calls, request.state.get('run_model_call_count', 0))
        async def attempt(current):
            if self.calls >= self.run_limit:
                raise ModelCallLimitExceededError(self.calls, self.calls, None, self.run_limit)
            self.calls += 1
            started = time.monotonic()
            final_only = self.reserve_final and current is not None and self.calls == self.run_limit
            if final_only:
                current = current.override(
                    tools=[], tool_choice='none',
                    messages=[*current.messages, HumanMessage(content=(
                        'This is the final request within the existing budget. Tool work has stopped. '
                        'Give the final report now using existing evidence: actual changes, checks that '
                        'passed or failed, and unfinished or unverified requirements. '
                        'Do not claim completion from passing tests alone. Do not request tools.'))])
            elif self.reserve_final and current is not None and self.run_limit - self.calls < 5:
                remaining = self.run_limit - self.calls + 1
                current = current.override(messages=[*current.messages, HumanMessage(content=(
                    f'Budget checkpoint: {remaining} model requests remain including this one; '
                    'the final request is reserved for reporting without tools. '
                    'Prioritize repairing observed failures and verifying the current required changes. '
                    'Avoid optional scope expansion; do not omit requirements, delete or weaken tests, '
                    'or repeat unchanged passing checks. If work remains, report it accurately.'))])
            record = {'attempt': self.calls, 'usage': None, 'final_only': final_only}
            try:
                response = await handler(current)
                messages = response.result if hasattr(response, 'result') else [response]
                record.update(outcome='returned', usage=messages[-1].usage_metadata)
                if final_only and messages[-1].tool_calls:
                    # A provider ignoring tool_choice must not execute one more action.
                    raise ModelCallLimitExceededError(self.calls, self.calls, None, self.run_limit)
                return response
            except BaseException as error:
                record.update(outcome=type(error).__name__,
                              status_code=getattr(error, 'status_code', None))
                raise
            finally:
                record['seconds'] = time.monotonic() - started
                self.records.append(record)
                if self.session:
                    with (self.session.directory / 'model-attempts.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(record, ensure_ascii=False) + '\n')
        return await self.retry.awrap_model_call(request, attempt)
