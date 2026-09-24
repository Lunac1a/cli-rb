"""Copy working-tree source without mounting the original repository."""

import hashlib
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

EXCLUDED = {
    ".git", ".cli-rb", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".idea", "dist", "build", ".next",
    ".ssh", ".aws", ".azure", ".kube", ".gnupg", "credentials", "secrets",
}
SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|"
    rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,}"
)


def excluded(path: PurePosixPath) -> bool:
    parts = [part.lower() for part in path.parts]
    name = parts[-1]
    return (
        any(part in EXCLUDED or part.startswith("pytest-cache-files-") for part in parts)
        or name.startswith(".env")
        or name in {".npmrc", ".pypirc", ".netrc", "id_rsa", "id_ed25519"}
        or name.startswith(("credentials.", "secrets."))
        or Path(name).suffix in {".pem", ".key", ".p12", ".pfx", ".keystore"}
    )


@dataclass(frozen=True)
class Snapshot:
    session: Path
    workspace: Path
    manifest: dict
    mode: str = "isolated"
    resumed: bool = False


def load_snapshot(directory: Path) -> Snapshot:
    session = directory.resolve(strict=True)
    manifest = json.loads((session / 'manifest.json').read_text(encoding='utf-8'))
    mode = manifest.get('mode')
    if manifest.get('status') != 'ready' or mode not in {'direct', 'isolated'}:
        raise ValueError('会话基线或模式不支持恢复。')
    workspace = Path(manifest['source']) if mode == 'direct' else session / 'workspace'
    if not workspace.is_dir():
        raise ValueError('原工作区不存在，拒绝恢复。')
    from cli_rb.patches import file_bytes
    baseline = session / ('workspace' if mode == 'direct' else 'baseline')
    for name, digest in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('会话基线路径无效。')
        if hashlib.sha256(file_bytes(baseline, name)).hexdigest() != digest:
            raise ValueError('会话起点基线已改变，拒绝恢复。')
    return Snapshot(session, workspace.resolve(), manifest, mode, resumed=True)


def create_snapshot(source: Path, sessions: Path) -> Snapshot:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("请选择 Git 仓库目录。")
    root = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        capture_output=True, check=True, timeout=10,
    ).stdout.decode("utf-8").strip()
    if Path(root).resolve() != source:
        raise ValueError("请选择 Git 仓库根目录。")
    entries = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True, check=True, timeout=10,
    ).stdout.decode("utf-8").split("\0")
    entries = sorted(set(filter(None, entries)))
    if len(entries) > 10000:
        raise ValueError("仓库超过 10000 个候选文件，本阶段不支持。")
    session = sessions.resolve() / uuid4().hex
    workspace = session / "workspace"
    workspace.mkdir(parents=True)
    manifest = {"source": str(source), "files": {}, "excluded": {}, "status": "preparing"}
    total = 0
    try:
        for name in entries:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("仓库包含不支持的路径。")
            reason = None
            if excluded(relative):
                reason = "excluded path"
            file = source.joinpath(*relative.parts)
            if not reason:
                current = source
                for part in relative.parts:
                    current /= part
                    if not current.exists() and not current.is_symlink():
                        reason = "deleted in working tree"
                        break
                    info = current.lstat()
                    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                        reason = "symlink or reparse point"
                        break
                if not reason and not file.is_file():
                    reason = "not a regular file (e.g. submodule)"
            if reason:
                manifest["excluded"][name] = reason
                continue
            # No source execution. Read bounded bytes and write new independent files.
            with file.open("rb") as stream:
                content = stream.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                manifest["excluded"][name] = "larger than 1 MiB"
                continue
            if SECRET.search(content):
                manifest["excluded"][name] = "recognized credential content"
                continue
            total += len(content)
            if total > 64 * 1024 * 1024:
                raise ValueError("工作副本超过 64 MiB，已停止准备。")
            target = workspace.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o644)
            manifest["files"][name] = hashlib.sha256(content).hexdigest()
        manifest["status"] = "ready"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        (session / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return Snapshot(session, workspace, manifest)


def create_session(source: Path, sessions: Path, mode: str = "direct") -> Snapshot:
    """Keep the filtered start snapshot as baseline; direct mode tools use the live root."""
    if mode not in {"direct", "isolated"}:
        raise ValueError("模式必须为 direct 或 isolated。")
    initial = create_snapshot(source, sessions)
    initial.manifest["mode"] = mode
    (initial.session / "manifest.json").write_text(
        json.dumps(initial.manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return Snapshot(initial.session, source.resolve() if mode == "direct" else initial.workspace,
                    initial.manifest, mode)
