"""Internal SWE-bench file boundary; no model invocation or scoring here.

Run with python -m cli_rb.evaluation.tasks. Raw datasets stay outside the solver
workspace. Only the explicit public task projection may be passed to a solver.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class SolverTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict):
            raise ValueError("任务必须是 JSON 对象。")
        fields = {}
        for name in cls.__dataclass_fields__:
            value = record.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"任务字段缺失或无效：{name}")
            fields[name] = value
        if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", fields["instance_id"]):
            raise ValueError("无效的 SWE-bench instance_id。")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", fields["repo"]):
            raise ValueError("无效的仓库标识。")
        if not re.fullmatch(r"[0-9a-f]{40}", fields["base_commit"]):
            raise ValueError("base_commit 必须为完整的小写 Git SHA。")
        # No raw record retained: patch, test_patch, test lists and future fields
        # cannot leak through serialization or a generic prompt formatter.
        return cls(**fields)


def select_tasks(source: Path, instance_ids: list[str]) -> list[SolverTask]:
    """Select explicit IDs from a local JSONL dataset, preserving caller order."""
    if not instance_ids or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("需要非空且不重复的任务 ID。")
    selected = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("数据集行必须是 JSON 对象。")
            instance_id = record.get("instance_id")
            if instance_id not in instance_ids:
                continue
            if instance_id in selected:
                raise ValueError("数据集包含重复任务 ID。")
            selected[instance_id] = SolverTask.from_record(record)
    if set(selected) != set(instance_ids):
        raise ValueError("数据集中缺少指定任务。")
    return [selected[instance_id] for instance_id in instance_ids]


def write_jsonl(path: Path, records):
    # Validate/serialize everything before creating the file; never overwrite
    # an earlier attempt, including an empty prediction.
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def export_prediction(task: SolverTask, model: str, patch: Path, output: Path):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("必须记录实际使用的模型标识。")
    # Preserve CRLF and no-newline markers from the existing PatchService.
    # Empty patches are valid submissions, not missing attempts or successes.
    model_patch = patch.read_bytes().decode("utf-8")
    write_jsonl(output, [{
        "instance_id": task.instance_id,
        "model_name_or_path": model,
        "model_patch": model_patch,
    }])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="导出仅含公开输入的任务文件")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--instance-id", action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("prediction", help="将已有补丁转成官方预测格式")
    prediction.add_argument("--tasks", type=Path, required=True)
    prediction.add_argument("--instance-id", required=True)
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--patch", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            tasks = select_tasks(args.dataset, args.instance_id)
            write_jsonl(args.output, [asdict(task) for task in tasks])
        else:
            task, = select_tasks(args.tasks, [args.instance_id])
            export_prediction(task, args.model, args.patch, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"评测文件处理失败：{error}\n")
    print(f"已写入 {args.output}；尚未运行求解或官方评分。")


if __name__ == "__main__":
    main()
