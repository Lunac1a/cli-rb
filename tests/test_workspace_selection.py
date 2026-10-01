from io import StringIO
from types import SimpleNamespace
import subprocess

import pytest
from rich.console import Console

from cli_rb.interface import workspace_selection as selection
from cli_rb.workspace.snapshot import create_snapshot


def setup_input(monkeypatch, paths, decisions):
    paths, decisions = iter(paths), iter(decisions)
    monkeypatch.setattr(selection, 'PromptSession', lambda: SimpleNamespace(
        prompt=lambda *a, **kw: str(next(paths))))
    monkeypatch.setattr(selection.typer, 'confirm', lambda *a, **kw: next(decisions))
    return Console(file=StringIO())


def test_ordinary_folder_initializes_and_creates_snapshot(tmp_path, monkeypatch):
    source = tmp_path / '普通项目'
    source.mkdir()
    (source / 'app.py').write_text('print(42)', encoding='utf-8')
    console = setup_input(monkeypatch, [source], [True])
    assert selection.select_workspace(console, '') == source.resolve()
    assert (source / '.git').is_dir()
    snapshot = create_snapshot(source, tmp_path / 'sessions')
    assert (snapshot.workspace / 'app.py').read_text() == 'print(42)'
    assert selection.git(source, 'rev-parse', '--verify', 'HEAD').returncode != 0


def test_decline_leaves_folder_untouched_and_reselects(tmp_path, monkeypatch):
    ordinary, repo = tmp_path / 'ordinary', tmp_path / 'repo'
    ordinary.mkdir()
    repo.mkdir()
    selection.git(repo, 'init')
    console = setup_input(monkeypatch, [tmp_path / 'missing', ordinary, repo], [False])
    assert selection.select_workspace(console, '') == repo.resolve()
    assert not (ordinary / '.git').exists()


def test_subdirectory_requires_confirmation(tmp_path, monkeypatch):
    selection.git(tmp_path, 'init')
    child = tmp_path / 'child'
    child.mkdir()
    console = setup_input(monkeypatch, [child, child], [False, True])
    assert selection.select_workspace(console, '') == tmp_path.resolve()


def test_other_git_errors_do_not_offer_initialization(tmp_path, monkeypatch):
    console = setup_input(monkeypatch, [tmp_path], [])
    monkeypatch.setattr(selection, 'git', lambda *a: subprocess.CompletedProcess(
        a, 128, '', 'fatal: detected dubious ownership'))
    with pytest.raises(StopIteration):
        selection.select_workspace(console, '')
    assert 'dubious ownership' in console.file.getvalue()
    assert not (tmp_path / '.git').exists()
