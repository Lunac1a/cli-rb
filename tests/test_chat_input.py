import sys

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cli_rb.interface.input import chat_bindings


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows console events')
def test_windows_enter_sends_and_shift_enter_keeps_multiline():
    from cli_rb.interface.input import ChatConsoleReader
    from prompt_toolkit.win32_types import KEY_EVENT_RECORD
    reader = object.__new__(ChatConsoleReader)
    event = KEY_EVENT_RECORD()
    event.KeyDown = 1
    event.VirtualKeyCode = 13
    event.uChar.UnicodeChar = '\r'
    enter = reader._event_to_key_presses(event)[0]
    event.ControlKeyState = reader.SHIFT_PRESSED
    shifted = reader._event_to_key_presses(event)[0]
    with create_pipe_input() as stream:
        session = PromptSession(input=stream, output=DummyOutput(), multiline=True,
                                key_bindings=chat_bindings())
        stream.send_text('first' + shifted.data + 'second' + enter.data)
        assert session.prompt() == 'first\nsecond'
        stream.send_text('next' + enter.data)
        assert session.prompt() == 'next'


def test_pasted_multiline_waits_for_enter():
    with create_pipe_input() as stream:
        session = PromptSession(input=stream, output=DummyOutput(), multiline=True,
                                key_bindings=chat_bindings())
        stream.send_text('\x1b[200~first\nsecond\x1b[201~ edited\r')
        assert session.prompt() == 'first\nsecond edited'
