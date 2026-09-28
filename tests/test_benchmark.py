import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from cli_rb.evaluation.tasks import SolverTask, export_prediction, select_tasks, write_jsonl
from cli_rb.workspace.patches import text_patch


def record():
    return dict(instance_id="demo__repo-1", repo="demo/repo", base_commit="a" * 40,
                problem_statement="修复 greeting", patch="GOLD_SECRET",
                test_patch="TEST_SECRET", FAIL_TO_PASS=["HIDDEN_TEST"],
                hints_text="HINT_SECRET", future_field="FUTURE_SECRET")


def test_prepare_cli_drops_all_nonpublic_fields(tmp_path):
    raw, public = tmp_path / "raw.jsonl", tmp_path / "public.jsonl"
    write_jsonl(raw, [record()])
    result = subprocess.run([
        sys.executable, "-m", "cli_rb.evaluation.tasks", "prepare", "--dataset", str(raw),
        "--instance-id", "demo__repo-1", "--output", str(public),
    ], capture_output=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(public.read_text(encoding="utf-8"))
    assert set(data) == {"instance_id", "repo", "base_commit", "problem_statement"}
    assert "SECRET" not in public.read_text(encoding="utf-8")
    assert "HIDDEN_TEST" not in public.read_text(encoding="utf-8")


def test_prediction_roundtrip_applies_exact_patch(tmp_path):
    task = SolverTask.from_record(record())
    patch, output = tmp_path / "change.patch", tmp_path / "prediction.jsonl"
    before = {"greeting.py": b"old\r\n"}
    after = {"greeting.py": b"new\r\n", "added.txt": b"no newline"}
    patch.write_bytes(text_patch(before, after).encode("utf-8"))
    export_prediction(task, "local-test-model", patch, output)
    prediction = json.loads(output.read_text(encoding="utf-8"))
    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert prediction["model_patch"].encode("utf-8") == patch.read_bytes()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "greeting.py").write_bytes(before["greeting.py"])
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"],
                             input=prediction["model_patch"].encode(), cwd=repo,
                             capture_output=True)
    assert applied.returncode == 0, applied.stderr
    assert {p.name: p.read_bytes() for p in repo.iterdir()} == after
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_prediction(task, "other", patch, output)
    assert output.read_bytes() == original


def test_empty_patch_is_preserved_without_success_claim(tmp_path):
    patch, output = tmp_path / "empty.patch", tmp_path / "prediction.jsonl"
    patch.write_bytes(b"")
    export_prediction(SolverTask.from_record(record()), "local", patch, output)
    assert json.loads(output.read_text())["model_patch"] == ""
    assert "resolved" not in json.loads(output.read_text())


def test_missing_duplicate_and_invalid_tasks_fail(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [record(), record()])
    with pytest.raises(ValueError, match="重复"):
        select_tasks(raw, ["demo__repo-1"])
    with pytest.raises(ValueError, match="缺少"):
        select_tasks(raw, ["demo__repo-2"])
    with pytest.raises(ValueError, match="不重复"):
        select_tasks(raw, ["demo__repo-1", "demo__repo-1"])
    bad = record() | {"base_commit": "main"}
    with pytest.raises(ValueError, match="SHA"):
        SolverTask.from_record(bad)
    assert "patch" not in asdict(SolverTask.from_record(record()))
