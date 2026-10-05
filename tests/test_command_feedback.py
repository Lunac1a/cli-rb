import json
import subprocess

import pytest

from cli_rb.workspace.command_feedback import command_feedback
from cli_rb.workspace.local_backend import LocalCodingBackend
from cli_rb.workspace.snapshot import create_session


def test_long_command_retains_tail_and_structured_evidence():
    original = {'exit_code': 1, 'timed_out': False,
                'output': 'begin\n' + 'x' * 20000 + '\nModuleNotFoundError: project_lib',
                'junit_xml': '<testsuite>' + 'x' * 20000 + '</testsuite>'}
    raw = json.dumps(original)
    feedback = json.loads(command_feedback(raw, 'cli-rb-' + 'a' * 32))
    assert feedback['exit_code'] == 1 and feedback['output_truncated']
    assert feedback['output'].startswith('begin')
    assert feedback['output'].endswith('ModuleNotFoundError: project_lib')
    assert len(feedback['junit_xml']) < 100
    assert json.loads(raw) == original


@pytest.fixture
def backend(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'check.py').write_text("print('x' * 20000)\nraise RuntimeError('missing project profile')\n")
    return LocalCodingBackend(create_session(repo, tmp_path / 'sessions'), approve=lambda _: True)


def test_failed_verification_exposes_diagnostic_and_lossless_log(backend):
    receipt = json.loads(backend.run_verification('python check.py', 'project checks', kind='check'))
    evidence = receipt['command_result']
    assert receipt['outcome'] == 'failed'
    assert evidence['started'] and evidence['initial_cwd'] == str(backend.snapshot.workspace)
    assert 'RuntimeError: missing project profile' in evidence['output']
    full = json.loads((backend.logs / (receipt['log_id'] + '.txt')).read_text(encoding='utf-8'))
    assert 'x' * 20000 in full['output']
    assert len(evidence['output']) < len(full['output'])


def test_denied_verification_records_no_execution(backend):
    backend.approve = lambda _: False
    receipt = json.loads(backend.run_verification('Set-Content denied.txt no', 'denied check', kind='check'))
    assert receipt['command_result']['denied']
    assert receipt['command_result']['started'] is False
    assert receipt['exit_code'] is None
    assert not (backend.snapshot.workspace / 'denied.txt').exists()
