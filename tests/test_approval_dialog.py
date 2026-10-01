import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cli_rb.interface.approval import approval_dialog


class ScreenOutput(DummyOutput):
    entered = 0
    restored = 0

    def enter_alternate_screen(self):
        self.entered += 1

    def quit_alternate_screen(self):
        self.restored += 1


@pytest.mark.parametrize('key,expected', [('y', True), ('n', False), ('\x04', EOFError)])
def test_approval_restores_screen_after_decision_or_close(key, expected):
    async def run():
        output = ScreenOutput()
        with create_pipe_input() as stream:
            app = approval_dialog('\n'.join(f'完整命令行 {i}' for i in range(100)),
                                  input=stream, output=output)
            stream.send_text(key)
            if expected is EOFError:
                with pytest.raises(EOFError):
                    await app.run_async()
            else:
                assert await app.run_async() is expected
        assert output.entered == 1 and output.restored == 1
    asyncio.run(run())
