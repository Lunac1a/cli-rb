"""Chat input: Enter sends; Shift+Enter inserts a newline on Windows."""
import sys
from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.key_binding import KeyBindings


def chat_bindings():
    bindings = KeyBindings()

    @bindings.add('c-m')
    def send(event):
        event.current_buffer.validate_and_handle()

    @bindings.add('c-j')
    def newline(event):
        event.current_buffer.insert_text('\n')

    return bindings


if sys.platform == 'win32':
    from prompt_toolkit.input.win32 import ConsoleInputReader, Win32Input
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys

    class ChatConsoleReader(ConsoleInputReader):
        def _event_to_key_presses(self, ev):
            # VT input loses the Shift modifier on Enter; read console events instead.
            if ev.VirtualKeyCode == 13 and ev.ControlKeyState & self.SHIFT_PRESSED:
                return [KeyPress(Keys.ControlJ, '\n')]
            return super()._event_to_key_presses(ev)

    class ChatWindowsInput(Win32Input):
        def __init__(self):
            super().__init__()
            self.console_input_reader.close()
            self.console_input_reader = ChatConsoleReader()
            self._use_virtual_terminal_input = False


@contextmanager
def chat_prompt(metrics=None):
    input_stream = ChatWindowsInput() if sys.platform == 'win32' else create_input()
    try:
        yield PromptSession(
            input=input_stream, multiline=True, key_bindings=chat_bindings(),
            bottom_toolbar=lambda: ' Enter 发送 · Shift+Enter 换行  |  ' + (metrics.text() if metrics else '就绪'),
        )
    finally:
        input_stream.close()
