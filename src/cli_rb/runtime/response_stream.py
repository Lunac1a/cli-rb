"""Preserve completed tool arguments when a Responses gateway omits deltas."""
import json
import re

import httpx2


class ToolArgumentStreamError(ValueError):
    pass


class IncompleteResponseStreamError(ValueError):
    """No completed response was returned; partial tools must not be dispatched."""


class ToolArgumentFrames:
    def __init__(self):
        self.pending = b''
        self.arguments = {}
        self.completed = False

    def finish(self):
        if not self.completed:
            raise IncompleteResponseStreamError('响应流缺少完成事件；部分响应未交给工具执行。')

    def feed(self, chunk):
        self.pending += chunk
        while separator := re.search(br'\r?\n\r?\n', self.pending):
            frame, self.pending = self.pending[:separator.end()], self.pending[separator.end():]
            yield self.repair(frame)

    def repair(self, frame):
        data = b'\n'.join(line[5:].lstrip(b' ') for line in frame.splitlines()
                          if line.startswith(b'data:'))
        if data == b'[DONE]':
            self.finish()
            return frame
        if not data:
            return frame
        event = json.loads(data)
        kind, item = event.get('type'), event.get('item', {})
        if kind == 'response.completed':
            self.completed = True
        if kind == 'response.output_item.added' and item.get('type') == 'function_call':
            self.arguments[item['id']] = item.get('arguments', '')
        elif kind == 'response.function_call_arguments.delta':
            key = event['item_id']
            self.arguments[key] = self.arguments.get(key, '') + event['delta']
        elif kind == 'response.function_call_arguments.done' or (
                kind == 'response.output_item.done' and item.get('type') == 'function_call'):
            key = event.get('item_id') or item['id']
            complete = event.get('arguments') if 'arguments' in event else item['arguments']
            seen = self.arguments.get(key)
            if seen is None or not complete.startswith(seen):
                raise ToolArgumentStreamError(
                    '工具参数流与完成记录不一致；未猜测参数或执行工具。')
            self.arguments[key] = complete
            if suffix := complete[len(seen):]:
                delta = {'type': 'response.function_call_arguments.delta',
                         'item_id': key, 'output_index': event['output_index'],
                         'sequence_number': event['sequence_number'], 'delta': suffix}
                return b'data: ' + json.dumps(delta).encode() + b'\n\n' + frame
        return frame


class AsyncToolArgumentStream(httpx2.AsyncByteStream):
    def __init__(self, stream):
        self.stream = stream
        self.frames = ToolArgumentFrames()

    async def __aiter__(self):
        async for chunk in self.stream:
            for frame in self.frames.feed(chunk):
                yield frame
        if self.frames.pending:
            yield self.frames.repair(self.frames.pending)
        self.frames.finish()

    async def aclose(self):
        await self.stream.aclose()


class ToolArgumentStream(httpx2.SyncByteStream):
    def __init__(self, stream):
        self.stream = stream
        self.frames = ToolArgumentFrames()

    def __iter__(self):
        for chunk in self.stream:
            yield from self.frames.feed(chunk)
        if self.frames.pending:
            yield self.frames.repair(self.frames.pending)
        self.frames.finish()

    def close(self):
        self.stream.close()


def install_tool_argument_repair(model):
    def streamed_response(response):
        return (response.request.url.path.endswith('/responses')
                and response.headers.get('content-type', '').startswith('text/event-stream'))

    def repair_sync(response):
        if streamed_response(response):
            response.stream = ToolArgumentStream(response.stream)

    async def repair_async(response):
        if streamed_response(response):
            response.stream = AsyncToolArgumentStream(response.stream)

    model.root_client._client.event_hooks['response'].append(repair_sync)
    model.root_async_client._client.event_hooks['response'].append(repair_async)
    return model
