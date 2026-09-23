"""Opt-in real Docker checks; unavailable Docker is a failure when requested."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from cli_rb.docker_backend import DockerReadBackend, prepare_image
from cli_rb.snapshot import create_snapshot
from cli_rb.coding_backend import DockerCodingBackend

pytestmark = pytest.mark.skipif(
    os.getenv("CLI_RB_DOCKER_TESTS") != "1", reason="set CLI_RB_DOCKER_TESTS=1 for real Docker",
)


def test_real_docker_read_only_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=excluded", encoding="utf-8")
    docker, image = prepare_image()
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    backend = DockerReadBackend(snapshot, docker, image)
    assert "app.py" in backend.list_files()
    assert "app.py:1:" in backend.search_files("greet")
    assert "return 'hello'" in backend.read_file("app.py")
    assert backend.read_file("../secret").startswith("工具错误")
    assert backend.read_file(".env").startswith("工具错误")
    # Direct OS-level probes using exactly the same run restrictions as the backend.
    command = backend.command("cli-rb-test-" + uuid4().hex)
    command[-1] = '''import json, os, socket
from pathlib import Path
checks = {"uid": os.getuid(), "no_docker_socket": not Path("/var/run/docker.sock").exists()}
try:
    Path("/workspace/app.py").write_text("changed")
    checks["write_denied"] = False
except OSError:
    checks["write_denied"] = True
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2).close()
    checks["network_denied"] = False
except OSError:
    checks["network_denied"] = True
status = Path("/proc/self/status").read_text()
checks["no_new_privileges"] = "NoNewPrivs:\\t1" in status
checks["no_capabilities"] = "CapEff:\\t0000000000000000" in status
print(json.dumps(checks))
'''
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    checks = json.loads(result.stdout)
    assert checks.pop("uid") == 65534
    assert all(checks.values()), checks
    assert (repo / "app.py").read_text() == "def greet():\n    return 'hello'\n"


def test_real_coding_edit_command_timeout_patch_and_boundary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    original = "def add(a, b):\n    return a - b\n"
    (repo / "app.py").write_text(original, encoding="utf-8")
    docker, image = prepare_image()
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    backend = DockerCodingBackend(snapshot, docker, image)
    failure = json.loads(backend.run_command('python -c "from app import add; assert add(2, 3) == 5"'))
    assert failure["exit_code"] != 0 and not failure["timed_out"]
    assert "已更新" in backend.edit_file("app.py", "return a - b", "return a + b")
    passed = json.loads(backend.run_command('python -c "from app import add; assert add(2, 3) == 5; print(123)"'))
    assert passed["exit_code"] == 0 and "123" in passed["output"]
    assert "已更新" in backend.create_file("new.txt", "new\n")
    assert "已删除" in backend.delete_file("new.txt", "new\n")
    assert "工具错误" in backend.create_file("../escape", "bad")
    timeout = json.loads(backend.run_command("sleep 10", timeout_seconds=1))
    assert timeout["timed_out"] and timeout["exit_code"] != 0
    assert "未应用" in backend.export_patch()
    assert (repo / "app.py").read_text() == original
    assert "return a + b" in backend.show_changes()
    # Shell cannot write outside mounted copy or find the protected baseline.
    denied = json.loads(backend.run_command("touch /cannot-write; test -e /baseline"))
    assert denied["exit_code"] != 0 and "Read-only file system" in denied["output"]
    # A command-created symlink must not be followed by host-side patch export.
    backend.run_command("ln -s /etc/passwd link")
    with pytest.raises(ValueError, match="链接"):
        backend.export_patch()


def test_cancel_keeps_writes_and_never_replays_command(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "readme").write_text("sample")
    docker, image = prepare_image()
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    backend = DockerCodingBackend(snapshot, docker, image)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.run_command, "echo saved > progress.txt; sleep 20")
        deadline = time.monotonic() + 10
        while not (snapshot.workspace / "progress.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (snapshot.workspace / "progress.txt").exists()
        backend.cancel_all()
        assert "工具错误" in future.result(timeout=10)
    assert (snapshot.workspace / "progress.txt").read_text().strip() == "saved"
    assert not (repo / "progress.txt").exists()
    with pytest.raises(RuntimeError, match="取消"):
        backend.run_command("echo replayed >> progress.txt")
    assert (snapshot.workspace / "progress.txt").read_text().strip() == "saved"


def test_direct_workspace_edits_permissions_and_masks(tmp_path):
    from cli_rb.snapshot import create_session
    repo = tmp_path / 'live'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'app.py').write_text('value = 1\n', encoding='utf-8')
    (repo / '.env').write_text('PRIVATE_VALUE=keep', encoding='utf-8')
    (repo / '.gitignore').write_text('private/\n', encoding='utf-8')
    (repo / 'private').mkdir()
    (repo / 'private/token').write_text('keep', encoding='utf-8')
    git_head = (repo / '.git/HEAD').read_bytes()
    snapshot = create_session(repo, tmp_path / 'sessions')
    backend = DockerCodingBackend(snapshot, *prepare_image())
    assert snapshot.workspace == repo.resolve()
    assert '已更新' in backend.edit_file('app.py', 'value = 1', 'value = 2')
    assert (repo / 'app.py').read_text() == 'value = 2\n'
    assert (snapshot.session / 'workspace/app.py').read_text() == 'value = 1\n'
    (repo / 'app.py').write_text('value = 3\n', encoding='utf-8')
    assert 'value = 3' in backend.read_file('app.py')
    assert '+value = 3' in backend.show_changes()
    result = json.loads(backend.run_command('echo bad > app.py'))
    assert result['exit_code'] != 0
    assert '授权未获得' in backend.run_write_command('echo bad > app.py')
    assert '授权未获得' in backend.delete_file('app.py', 'value = 3\n')
    assert '工具错误' in backend.read_file('.env')
    masks = json.loads(backend.run_command('test ! -s .env && test ! -e .git/HEAD && test ! -e private/token'))
    assert masks['exit_code'] == 0, masks
    assert (repo / '.env').read_text() == 'PRIVATE_VALUE=keep'
    assert (repo / '.git/HEAD').read_bytes() == git_head
    approvals = []
    backend.approve = lambda action: approvals.append(action) or True
    result = json.loads(backend.run_write_command('echo approved > result.txt'))
    assert result['exit_code'] == 0, result
    assert (repo / 'result.txt').read_text().strip() == 'approved'
    assert len(approvals) == 1
    assert '已更新' in backend.create_file('new.txt', 'new')
    assert '已删除' in backend.delete_file('new.txt', 'new')
    assert len(approvals) == 1
    assert '已删除' in backend.delete_file('app.py', 'value = 3\n')
    assert len(approvals) == 2
    assert '修改已在本地仓库' in backend.export_patch()
