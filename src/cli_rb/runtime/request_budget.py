"""Optional cumulative evaluation budget; incomplete request usage stays unknown."""
import json
import time

from langchain_core.callbacks import AsyncCallbackHandler
from cli_rb.runtime.response_stream import IncompleteResponseStreamError


class RequestBudgetExceeded(RuntimeError):
    pass


class PersistentRequestBudget(AsyncCallbackHandler):
    raise_error = True

    def __init__(self, path, *, seconds, requests, tokens, incomplete_token_bound=None):
        self.path = path
        limits = {'seconds': seconds, 'requests': requests, 'tokens': tokens}
        if incomplete_token_bound is not None:
            limits['incomplete_token_bound'] = incomplete_token_bound
        self.state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'started': time.time(), 'requests': 0, 'tokens': 0,
            'unknown_usage': False, 'inflight': False, 'limits': limits}
        if self.state['limits'] != limits:
            raise ValueError('累计预算已冻结，不能在恢复时更改。')
        if self.state['inflight']:
            self.state['unknown_usage'] = True
        self.save()

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    @property
    def remaining(self):
        return self.state['limits']['seconds'] - (time.time() - self.state['started'])

    def attach(self, model):
        """Meter sequential async LangChain calls, including its summarizer copies."""
        model.callbacks = [*(model.callbacks or []), self]
        model.root_async_client._client.event_hooks['request'].append(self.before_http)

    async def before_http(self, request):
        if (self.remaining <= 0 or self.state['requests'] >= self.state['limits']['requests']
                or self.state['tokens'] >= self.state['limits']['tokens']
                or self.state['unknown_usage'] or self.state['inflight']):
            raise RequestBudgetExceeded('累计预算耗尽或有请求用量未知；没有发起新请求。')
        self.state['requests'] += 1
        self.state['inflight'] = True
        self.save()

    async def on_llm_end(self, response, **kwargs):
        usage = response.generations[0][0].message.usage_metadata
        if usage is None or 'total_tokens' not in usage:
            self.state['unknown_usage'] = True
        else:
            self.state['tokens'] += usage['total_tokens']
        self.state['inflight'] = False
        self.save()
        with self.path.with_suffix('.requests.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'time': time.time(), 'usage': usage}) + '\n')

    async def on_llm_error(self, error, **kwargs):
        if self.state['inflight']:
            bound = self.state['limits'].get('incomplete_token_bound')
            if isinstance(error, IncompleteResponseStreamError) and bound is not None:
                self.state['tokens'] += bound
                self.state['bounded_unknown_tokens'] = self.state.get('bounded_unknown_tokens', 0) + bound
                self.state['bounded_unknown_requests'] = self.state.get('bounded_unknown_requests', 0) + 1
                self.state['inflight'] = False
                with self.path.with_suffix('.requests.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps({'time': time.time(), 'usage': None,
                                          'error': type(error).__name__, 'token_upper_bound': bound}) + '\n')
            else:
                self.state['unknown_usage'] = True
            self.save()
