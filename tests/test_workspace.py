import json
import subprocess

import pytest

from cli_rb.container_worker import execute
from cli_rb.docker_backend import DockerReadBackend
from cli_rb.snapshot import Snapshot, create_snapshot


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_snapshot_includes_uncommitted_and_untracked_but_excludes_secrets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    (repo / "app.py").write_text("old", encoding="utf-8")
    (repo / ".env").write_text("SECRET=do-not-copy", encoding="utf-8")
    git(repo, "add", "app.py", ".env")
    (repo / "app.py").write_text("uncommitted", encoding="utf-8")
    (repo / "new.py").write_text("new", encoding="utf-8")
    (repo / "token.txt").write_text("ghp_" + "x" * 36, encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules/a.js").write_text("dependency")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text("ignored")
    snap = create_snapshot(repo, tmp_path / "sessions")
    assert (snap.workspace / "app.py").read_text() == "uncommitted"
    assert (snap.workspace / "new.py").read_text() == "new"
    for name in [".env", "token.txt", "node_modules", ".git", "ignored.txt"]:
        assert not (snap.workspace / name).exists()
    assert ".env" in snap.manifest["excluded"]
    assert json.loads((snap.session / "manifest.json").read_text())["status"] == "ready"
    (snap.workspace / "app.py").write_text("copy-only")
    assert (repo / "app.py").read_text() == "uncommitted"


def test_worker_list_search_read_and_traversal(tmp_path):
    (tmp_path / "app.py").write_text("def hello():\n    return 42\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("hello again\n", encoding="utf-8")
    assert "app.py" in execute(tmp_path, {"action": "list", "args": {}})
    assert "offset=1" in execute(tmp_path, {"action": "list", "args": {"limit": 1}})
    assert "app.py:1:" in execute(tmp_path, {"action": "search", "args": {"query": "hello"}})
    assert execute(tmp_path, {"action": "read", "args": {"path": "app.py", "start_line": 2}}) == "2:     return 42"
    for path in ["../secret", "/etc/passwd", "C:/secret", "a\\b"]:
        with pytest.raises(ValueError):
            execute(tmp_path, {"action": "read", "args": {"path": path}})
    with pytest.raises(FileNotFoundError):
        execute(tmp_path, {"action": "read", "args": {"path": "missing"}})
    with pytest.raises(ValueError):
        execute(tmp_path, {"action": "list", "args": {"limit": 10000}})


def test_timeout_removes_only_its_container_without_retry(tmp_path, monkeypatch):
    backend = DockerReadBackend(Snapshot(tmp_path, tmp_path, {}), "docker", "sha256:" + "a" * 64)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "start":
            raise subprocess.TimeoutExpired(args, 30)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("cli_rb.docker_backend.subprocess.run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        backend.read_file("app.py")
    assert len(calls) == 4
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[2] == calls[3] == ["docker", "rm", "--force", name]
    assert list((tmp_path / "tool-logs").glob("*-interrupted.json"))


def test_binary_and_long_output_are_explicit(tmp_path):
    (tmp_path / "binary").write_bytes(b"a\0b")
    with pytest.raises(ValueError):
        execute(tmp_path, {"action": "read", "args": {"path": "binary"}})
    (tmp_path / "long").write_text("a" * 3000)
    assert "截断" in execute(tmp_path, {"action": "read", "args": {"path": "long"}})
