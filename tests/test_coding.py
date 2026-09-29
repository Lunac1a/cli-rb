import json
import subprocess
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from cli_rb.interface.cli import report_empty_reply
from cli_rb.workspace.container_worker import execute
from cli_rb.workspace.patches import PatchService, text_patch, inventory, file_bytes
from cli_rb.workspace.snapshot import create_snapshot


def test_edit_rejects_stale_or_ambiguous_content(tmp_path):
    file = tmp_path / "a.py"
    file.write_text("one\none\n", encoding="utf-8")
    with pytest.raises(ValueError):
        execute(tmp_path, {"action": "edit", "args": {
            "path": "a.py", "old_text": "one", "new_text": "two",
        }}, writable=True)
    assert file.read_text() == "one\none\n"
    execute(tmp_path, {"action": "edit", "args": {
        "path": "a.py", "old_text": "one\none", "new_text": "two",
    }}, writable=True)
    assert file.read_text() == "two\n"
    with pytest.raises(ValueError):
        execute(tmp_path, {"action": "delete", "args": {
            "path": "a.py", "expected_content": "old",
        }}, writable=True)
    execute(tmp_path, {"action": "delete", "args": {
        "path": "a.py", "expected_content": "two\n",
    }}, writable=True)
    assert not file.exists()


def test_readonly_and_traversal_do_not_write(tmp_path):
    request = {"action": "create", "args": {"path": "new", "content": "hi"}}
    with pytest.raises(ValueError):
        execute(tmp_path, request)
    for path in ["../escaped", "/tmp/escaped", ".git/config"]:
        with pytest.raises(ValueError):
            execute(tmp_path, {"action": "create", "args": {"path": path, "content": "bad"}}, writable=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("before,after", [
    ({"a.py": b"old\n", "delete.txt": b"bye\n"}, {"a.py": b"new\n", "new.txt": b"hello\n"}),
    ({"empty": b""}, {"newempty": b""}),
    ({"file with space.txt": b"old"}, {"file with space.txt": b"new"}),
    ({"中文.txt": b"old\r\n"}, {"中文.txt": b"new\r\n"}),
])
def test_patch_applies_exact_bytes_including_added_deleted_files(tmp_path, before, after):
    for name, content in before.items():
        (tmp_path / name).write_bytes(content)
    patch = tmp_path.parent / f"{tmp_path.name}.patch"
    patch.write_bytes(text_patch(before, after).encode("utf-8"))
    run = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch)],
                         cwd=tmp_path, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == after


def test_patch_rejects_changed_binary_or_known_secret():
    with pytest.raises(ValueError):
        text_patch({"binary": b"\0a"}, {"binary": b"\0b"})
    with pytest.raises(ValueError):
        text_patch({}, {"secret.txt": b"ghp_" + b"x" * 36})


def test_inventory_reads_chunk_boundaries_and_ignores_exact_subtrees(tmp_path):
    (tmp_path / 'kept').mkdir()
    (tmp_path / 'kept/a').write_bytes(b'x' * 65536)
    (tmp_path / 'kept/b').write_bytes(b'y' * 65537)
    (tmp_path / 'omit').mkdir()
    (tmp_path / 'omit/large').write_bytes(b'z' * 65538)
    assert inventory(tmp_path, ['omit'], max_file_bytes=65537) == {
        'kept/a': b'x' * 65536, 'kept/b': b'y' * 65537}
    assert file_bytes(tmp_path, 'kept/b', 65537) == b'y' * 65537
    with pytest.raises(ValueError, match='65536'):
        inventory(tmp_path, ['omit'], max_file_bytes=65536)
    with pytest.raises(ValueError, match='65536'):
        file_bytes(tmp_path, 'kept/b', 65536)


def test_empty_reply_reports_evidence_without_claiming_success(capsys):
    backend = SimpleNamespace(command_evidence=[{"command": "test", "exit_code": 1, "timed_out": False}],
                              patches=SimpleNamespace(last_export=None))
    assert report_empty_reply([AIMessage(content="")], backend)
    text = capsys.readouterr().out
    assert "未返回有效的最终文字答复" in text and '"exit_code": 1' in text
    assert "没有自动重跑" in text and "需求全部完成" in text
    assert not report_empty_reply([AIMessage(content="完成说明")], backend)


def test_baseline_is_not_original_head_and_export_is_application_owned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "a.py").write_text("uncommitted", encoding="utf-8")
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    service = PatchService(snapshot)
    (repo / "a.py").write_text("later user edit", encoding="utf-8")
    (snapshot.workspace / "a.py").write_text("agent edit", encoding="utf-8")
    diff = service.diff()
    assert "-uncommitted" in diff
    assert "+agent edit" in diff
    assert "later user edit" not in diff
    result = service.export()
    assert "未应用到原仓库" in result
    patch = next((snapshot.session / "exports").glob("*.patch"))
    assert patch.read_bytes().decode("utf-8") == diff
    assert not json.loads(patch.with_suffix(".json").read_text())["applied_to_source"]
    assert (repo / "a.py").read_text() == "later user edit"
