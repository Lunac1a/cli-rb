import json
from types import SimpleNamespace

from cli_rb.workspace.command_review import CommandReview, requires_human
from test_local_backend import backend as local_backend

backend = local_backend


def test_review_routing_keeps_high_impact_human():
    for command in ['git push origin main', 'Remove-Item -Recurse app', 'git reset --hard']:
        assert requires_human(command)
    assert not requires_human('Get-Location; Get-ChildItem | Select-Object Name')


def test_separate_review_logs_and_invalid_response_falls_to_human(tmp_path):
    reviewer = object.__new__(CommandReview)
    reviewer.calls, reviewer.directory = 0, tmp_path
    reviewer.model = SimpleNamespace(invoke=lambda _: SimpleNamespace(
        text='{"decision":"allow","reason":"项目内查看"}', usage_metadata={'total_tokens': 20}))
    assert reviewer.review('Get-Location; Get-ChildItem', '检查项目', tmp_path)[0]
    reviewer.model = SimpleNamespace(invoke=lambda _: SimpleNamespace(text='not json'))
    assert not reviewer.review('x', 'task', tmp_path)[0]
    rows = [json.loads(s) for s in (tmp_path/'approval-reviews.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[0]['usage']['total_tokens'] == 20 and rows[1]['decision'] == 'human'


def test_human_denial_never_runs_or_calls_reviewer(backend, monkeypatch):
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError('started denied command')))
    backend.reviewer = SimpleNamespace(review=lambda *a: (_ for _ in ()).throw(AssertionError('reviewed push')))
    backend.approve = lambda _: False
    receipt = json.loads(backend.run_command('git push origin main'))
    assert receipt['denied'] and receipt['started'] is False
    assert not backend.command_evidence
    saved = json.loads((backend.logs / (receipt['log_id'] + '.json')).read_text(encoding='utf-8'))
    assert saved['request']['args']['command'] == 'git push origin main'
    assert json.loads(saved['result'])['denied']
    approvals = (backend.snapshot.session / 'approvals.jsonl').read_text(encoding='utf-8')
    assert json.loads(approvals.splitlines()[-1])['accepted'] is False


def test_reviewed_compound_command_runs_directly_and_can_continue(backend, monkeypatch):
    import subprocess
    from cli_rb.runtime.persistence import SessionStore
    calls = []
    original = subprocess.Popen

    def launch(argv, **kwargs):
        calls.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', launch)
    reviewed = []
    backend.reviewer = SimpleNamespace(review=lambda command, *args:
        (reviewed.append(command) or True, '普通本地输出'))
    backend.approve = lambda _: (_ for _ in ()).throw(AssertionError('unnecessary human approval'))
    with SessionStore(backend.snapshot.session, create=True) as store:
        result = json.loads(backend.run_command("Write-Output 'one'; Write-Output 'two'"))
        assert result['exit_code'] == 0 and 'two' in result['output']
        store.require_idle()
    assert len(reviewed) == 1
    assert calls[0][0] == backend.shell and not any('codex' in str(arg).lower() for arg in calls[0])
