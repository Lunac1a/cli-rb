"""Identify paths to mask before binding a live workspace into Docker."""

import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

from cli_rb.snapshot import SECRET, excluded


def ignored_paths(root):
    run = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--others", "--ignored",
         "--exclude-standard", "--directory"], capture_output=True, check=True, timeout=10,
    )
    return {p.rstrip("/") for p in run.stdout.decode("utf-8").split("\0") if p}


def protected_paths(root: Path):
    ignored = ignored_paths(root)
    protected = {}
    examined = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs) + files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            examined += 1
            if examined > 10000:
                raise ValueError("工作区路径超过直接模式的检查上限。")
            info = path.lstat()
            linked = stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            if linked:
                # Docker nested mounts must never resolve a host link to an outside target.
                raise ValueError(f"直接模式不挂载含链接/重解析点的工作区：{relative}；可用隔离模式。")
            blocked = excluded(PurePosixPath(relative)) or relative in ignored
            if not blocked and stat.S_ISREG(info.st_mode):
                if info.st_size > 1024 * 1024:
                    blocked = True
                else:
                    with path.open("rb") as stream:
                        blocked = bool(SECRET.search(stream.read(1024 * 1024)))
            if blocked:
                protected[relative] = name in dirs
                if name in dirs:
                    dirs.remove(name)
    return protected
