import json
import os
import subprocess
import threading
import time

import pytest

from cli_rb.workspace.local_backend import LocalCodingBackend, project_environment
from cli_rb.workspace.snapshot import create_session
from cli_rb.runtime.prompt_loader import coding_prompt


@pytest.fixture
def backend(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'app.py').write_text('value = 1\n')
    return LocalCodingBackend(create_session(repo, tmp_path / 'sessions'), approve=lambda _: True)


def test_files_local_command_evidence_and_denial(backend):
    backend.create_file('hello.txt', 'hello')
    assert (backend.snapshot.workspace / 'hello.txt').read_text() == 'hello'
    result = json.loads(backend.run_command("Write-Output 'local-ok'"))
    assert result['exit_code'] == 0 and 'local-ok' in result['output']
    assert backend.command_evidence and not (backend.snapshot.session / 'container.json').exists()
    backend.approve = lambda _: False
    denied = json.loads(backend.run_command("Set-Content denied.txt 'bad'"))
    assert denied['denied'] and not (backend.snapshot.workspace / 'denied.txt').exists()


def test_timeout_and_cancellation_stop_running_command(backend):
    result = json.loads(backend.run_command('Start-Sleep -Seconds 20', 1))
    assert result['timed_out'] and not backend._active
    thread = threading.Thread(target=lambda: backend.run_command('Start-Sleep -Seconds 20'))
    thread.start()
    deadline = time.monotonic() + 5
    while not backend._active and time.monotonic() < deadline:
        time.sleep(0.02)
    assert backend._active
    backend.cancel_all()
    thread.join(5)
    assert not thread.is_alive() and not backend._active


def test_project_interpreter_priority_and_local_instructions(backend):
    directory = backend.snapshot.workspace / '.venv/Scripts'
    directory.mkdir(parents=True)
    (directory / 'python.exe').touch()
    env = project_environment(backend.snapshot.workspace)
    assert env['PATH'].split(os.pathsep)[0] == str(directory)
    prompt = coding_prompt(mode='direct', execution_rules=backend.execution_rules)
    assert 'NOT sandboxed' in prompt
    assert 'Commands run in a Linux container' not in prompt
    assert 'run_command is read-only' not in prompt


def test_local_verification_uses_real_exit_status(backend):
    receipt = json.loads(backend.run_verification('exit 7', 'failing check', kind='check'))
    assert receipt['outcome'] == 'failed'
