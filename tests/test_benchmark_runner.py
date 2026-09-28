import subprocess
from types import SimpleNamespace

import pytest

from cli_rb.evaluation.runner import BenchmarkBackend, verify_checkout
from cli_rb.evaluation.tasks import SolverTask
from cli_rb.workspace.snapshot import create_snapshot


def test_large_benchmark_fixture_survives_baseline_patch_and_reload(tmp_path):
    from cli_rb.workspace.patches import PatchService
    from cli_rb.workspace.snapshot import load_snapshot

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    data = b'\x00fixture' * 160000  # > 1 MiB, not editable source.
    (repo / 'sample.dat').write_bytes(data)
    (repo / 'app.py').write_text('before\n')
    ordinary = create_snapshot(repo, tmp_path / 'normal')
    assert 'sample.dat' in ordinary.manifest['excluded']
    snapshot = create_snapshot(repo, tmp_path / 'benchmark', max_file_bytes=64 * 1024 * 1024)
    patches = PatchService(snapshot)
    assert (snapshot.workspace / 'sample.dat').read_bytes() == data
    assert patches.diff() == ''
    (snapshot.workspace / 'app.py').write_text('after\n')
    patch = patches.diff()
    assert '+after' in patch and 'sample.dat' not in patch
    restored = load_snapshot(snapshot.session)
    assert PatchService(restored).diff() == patch


def test_benchmark_profile_retains_isolation_and_uses_environment(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.py').write_text('hello')
    snapshot = create_snapshot(repo, tmp_path / 'sessions')
    backend = BenchmarkBackend(snapshot, 'docker', 'sha256:' + 'a' * 64)
    command = backend.command('test-container')
    assert '--network=none' in command
    assert '--cap-drop=ALL' in command
    assert '--read-only' in command
    assert '--memory=2g' in command
    assert command[command.index('--user') + 1] == '65534:65534'
    assert '/opt/miniconda3/bin/python' in command
    assert not any('docker.sock' in x for x in command)


def test_checkout_gate_rejects_wrong_commit_and_dirty_state(tmp_path, monkeypatch):
    task = SolverTask('demo__repo-1', 'demo/repo', 'a' * 40, 'fix')
    responses = iter([b'b' * 40])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(responses)))
    with pytest.raises(ValueError, match='base_commit'):
        verify_checkout(tmp_path, task)
    responses = iter([b'a' * 40, b' M source.py'])
    with pytest.raises(ValueError, match='基线'):
        verify_checkout(tmp_path, task)
