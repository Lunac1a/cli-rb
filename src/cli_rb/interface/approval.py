"""Bridge synchronous tool approvals to cancellable terminal input."""

import asyncio
from concurrent.futures import CancelledError
import threading


class TerminalApproval:
    def __init__(self, loop, ask, cancel_turn):
        self.loop = loop
        self.ask = ask
        self.cancel_turn = cancel_turn
        self._lock = threading.Lock()
        self._pending = set()
        self._closed = False

    def __call__(self, action):
        # Called by a tool worker, never by the event-loop thread.
        with self._lock:
            if self._closed:
                return False
            future = asyncio.run_coroutine_threadsafe(self._ask(action), self.loop)
            self._pending.add(future)
        try:
            return future.result()
        except CancelledError:
            return False
        finally:
            with self._lock:
                self._pending.discard(future)

    async def _ask(self, action):
        try:
            return await self.ask(action)
        except KeyboardInterrupt:
            self.cancel_turn()
            return False
        except EOFError:
            return False

    def close(self):
        with self._lock:
            self._closed = True
            for future in self._pending:
                future.cancel()
