"""Application-owned baseline and text patches; never execute repository code on host."""

import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from uuid import uuid4

from cli_rb.snapshot import SECRET, Snapshot, excluded


def file_bytes(root: Path, name: str) -> bytes:
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("补丁不接受链接或重解析点。")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("补丁仅支持普通文件。")
    with current.open("rb") as stream:
        content = stream.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError(f"文件超过 1 MiB，无法导出：{name}")
    return content


def inventory(root: Path, ignored=()):
    files = {}
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs) + names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if excluded(PurePosixPath(relative)) or any(relative == p or relative.startswith(p + "/") for p in ignored):
                if name in dirs:
                    dirs.remove(name)
                continue
            if any(ord(c) < 32 for c in relative) or ":" in relative or "\\" in relative:
                raise ValueError("补丁包含不支持的文件名。")
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"发现链接，拒绝导出：{relative}")
            if name in dirs:
                continue
            data = file_bytes(root, relative)
            total += len(data)
            if len(files) >= 10000 or total > 64 * 1024 * 1024:
                raise ValueError("副本超过补丁大小限制。")
            files[relative] = data
    return files


def git_quote(path):
    # Git's C-style path quoting uses UTF-8 bytes and octal escapes, not JSON \u.
    return '"' + "".join(
        chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:03o}"
        for b in path.encode("utf-8")
    ) + '"'


def text_patch(before, after):
    chunks = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        if SECRET.search(new or b""):
            raise ValueError(f"变更包含已知凭据特征，拒绝导出：{name}")
        if b"\0" in (old or b"") + (new or b""):
            raise ValueError(f"本阶段不导出二进制变更：{name}")
        a, b = git_quote("a/" + name), git_quote("b/" + name)
        chunks.append(f"diff --git {a} {b}\n")
        if old is None:
            chunks.append("new file mode 100644\n")
        if new is None:
            chunks.append("deleted file mode 100644\n")
        lines = list(difflib.unified_diff(
            (old or b"").decode("utf-8").splitlines(keepends=True),
            (new or b"").decode("utf-8").splitlines(keepends=True),
            fromfile=a if old is not None else "/dev/null",
            tofile=b if new is not None else "/dev/null",
        ))
        for line in lines:
            chunks.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(chunks)


class PatchService:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.last_export = None
        if snapshot.mode == "direct":
            self.baseline = snapshot.session / "workspace"
            return
        self.baseline = snapshot.session / "baseline"
        self.baseline.mkdir(exist_ok=False)
        for name, digest in snapshot.manifest["files"].items():
            content = file_bytes(snapshot.workspace, name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("副本在准备基线前发生变化。")
            target = self.baseline / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def diff(self):
        before = {name: file_bytes(self.baseline, name) for name in self.snapshot.manifest["files"]}
        ignored = set(self.snapshot.manifest["excluded"])
        if self.snapshot.mode == "direct":
            from cli_rb.live_workspace import ignored_paths
            ignored |= ignored_paths(self.snapshot.workspace) - before.keys()
        return text_patch(before, inventory(self.snapshot.workspace, ignored))

    def export(self):
        patch = self.diff()
        if not patch:
            return "没有变更，未生成补丁。"
        directory = self.snapshot.session / "exports"
        directory.mkdir(exist_ok=True)
        path = directory / f"changes-{uuid4().hex}.patch"
        path.write_bytes(patch.encode("utf-8"))
        (path.with_suffix(".json")).write_text(json.dumps({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "base": "filtered session-start working tree (including uncommitted changes)",
            "workspace_mode": self.snapshot.mode,
            "changes_already_in_source": self.snapshot.mode == "direct",
            "applied_to_source": False,
        }, indent=2), encoding="utf-8")
        self.last_export = path
        status = "修改已在本地仓库中；导出仅用于保存差异。" if self.snapshot.mode == "direct" else "未应用到原仓库。"
        return f"补丁已导出：{path}\n相对于本次会话起点；{status}"
