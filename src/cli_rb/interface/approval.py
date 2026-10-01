"""Bridge synchronous tool approvals to cancellable terminal input."""

import asyncio
from concurrent.futures import CancelledError
import threading

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit
from prompt_toolkit.widgets import Button, Frame, Label, TextArea
from prompt_toolkit.styles import Style


def approval_dialog(action, *, input=None, output=None):
    """Use an alternate screen so long commands never remain in chat scrollback."""
    bindings = KeyBindings()

    @bindings.add('y')
    @bindings.add('Y')
    def allow(event):
        event.app.exit(result=True)

    @bindings.add('n')
    @bindings.add('N')
    @bindings.add('escape')
    def deny(event):
        event.app.exit(result=False)

    @bindings.add('c-c')
    def cancel(event):
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add('c-d')
    def close(event):
        event.app.exit(exception=EOFError())

    details = TextArea(text=action, read_only=True, scrollbar=True, wrap_lines=True)
    app = Application(
        layout=Layout(HSplit([
            Frame(details, title='需要你的授权'),
            Label('↑↓ / PgUp PgDn 查看完整操作 · Y 允许 · N / Esc 拒绝 · Ctrl+C 停止任务'),
            VSplit([Button('允许', handler=lambda: app.exit(result=True)),
                    Button('拒绝', handler=lambda: app.exit(result=False))], padding=3),
        ]), focused_element=details),
        key_bindings=bindings, full_screen=True, erase_when_done=True,
        mouse_support=True, input=input, output=output,
        style=Style.from_dict({'frame.border': 'ansiyellow', 'frame.label': 'bold ansiyellow'}),
    )
    return app


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
