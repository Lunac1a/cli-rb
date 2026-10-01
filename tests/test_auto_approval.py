import json

import pytest

from cli_rb.workspace.approval_policy import command_decision
from test_local_backend import backend as local_backend

backend = local_backend


@pytest.mark.parametrize('command', ['python -m pytest tests', 'python -m unittest discover',
                                    'node --test tests/stats.test.js', 'git diff', 'Get-Content app.py'])
def test_routine_commands(command):
    assert command_decision(command)[0]


@pytest.mark.parametrize('command', ['git push origin main', 'git reset --hard', 'Remove-Item -Recurse .',
    'python -c "import shutil"', 'git status; git push', 'Get-Content $(Get-Location)',
    'Get-Content ../secret', 'Get-Content .env.local', 'Get-Content *', 'npm install -g foo'])
def test_significant_or_unclear_commands_require_review(command):
    assert not command_decision(command)[0]


def test_automatic_execution_and_denied_destructive_command_are_logged(backend):
    questions = []
    backend.approve = lambda message: questions.append(message) or False
    result = json.loads(backend.run_command("Write-Output 'routine'"))
    assert result['exit_code'] == 0 and not questions
    result = json.loads(backend.run_command('Remove-Item app.py'))
    assert result['denied'] and len(questions) == 1
    assert (backend.snapshot.workspace / 'app.py').exists()
    records = [json.loads(line) for line in (backend.snapshot.session / 'approvals.jsonl').read_text(encoding='utf-8').splitlines()]
    assert records[0]['automatic'] and records[0]['accepted']
    assert not records[1]['automatic'] and not records[1]['accepted']


def test_backed_up_small_deletion_is_automatic_but_unbacked_edit_requires_review(backend):
    backend.approve = lambda _: pytest.fail('Unexpected confirmation for backed-up file')
    backend.delete_file('app.py', 'value = 1\n')
    assert not (backend.snapshot.workspace / 'app.py').exists()
    (backend.snapshot.workspace / 'app.py').write_text('valuable new content')
    backend.approve = lambda _: False
    assert '授权未获得' in backend.delete_file('app.py', 'valuable new content')
    assert (backend.snapshot.workspace / 'app.py').read_text() == 'valuable new content'
