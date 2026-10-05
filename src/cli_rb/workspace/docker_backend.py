"""A read-only container per tool call. No host execution fallback."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from uuid import uuid4

from cli_rb.workspace.snapshot import Snapshot
from cli_rb.workspace.command_feedback import command_feedback

IMAGE = "python:3.12-slim-bookworm"


def ensure_session_stopped(directory: Path, docker: str):
    """A crashed host can leave a container alive; never recover over it."""
    names = {p.name.removesuffix('-action.json') for p in (directory / 'tool-logs').glob('*-action.json')}
    result = subprocess.run([docker, 'ps', '-a', '--format', '{{.Names}}'],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('无法确认旧容器已结束，拒绝继续。')
    remaining = names.intersection(result.stdout.splitlines())
    if remaining:
        raise RuntimeError('旧会话仍有容器存在，先停止并清理后再恢复：' + ', '.join(sorted(remaining)))


def docker_executable():
    found = shutil.which("docker")
    if found:
        return found
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker/resources/bin/docker.exe"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("未找到 Docker，请先安装并启动 Docker Desktop。")


def prepare_image(image_name=None):
    docker = docker_executable()
    image_name = image_name or os.getenv("CLI_RB_IMAGE", IMAGE)
    if not image_name or image_name.startswith("-") or any(c.isspace() for c in image_name):
        raise ValueError("无效的 CLI_RB_IMAGE。")
    check = subprocess.run(
        [docker, "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=15,
    )
    if check.returncode or check.stdout.strip() != "linux":
        raise RuntimeError("Docker Linux 引擎未就绪；不会改用宿主机执行。")
    inspect = subprocess.run(
        [docker, "image", "inspect", image_name, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=15,
    )
    if inspect.returncode:
        raise RuntimeError(f"准备镜像缺失：{image_name}。请提前拉取或构建；工具执行期间不会下载镜像。")
    image = inspect.stdout.strip()
    if not image.startswith("sha256:") or len(image) != 71:
        raise RuntimeError("无法确认本地镜像 ID。")
    return docker, image


class DockerReadBackend:
    def __init__(self, snapshot: Snapshot, docker: str, image: str, *, writable=False):
        self.snapshot = snapshot
        self.docker = docker
        self.image = image
        self.writable = writable
        self._lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._active = set()
        self._cancelled = threading.Event()
        self.command_evidence = []
        self._mount_writable = writable
        self._protected = {}
        self.worker = Path(__file__).with_name("container_worker.py").read_text(encoding="utf-8")
        self.logs = snapshot.session / "tool-logs"
        self.logs.mkdir(exist_ok=True)
        (snapshot.session / "container.json").write_text(
            json.dumps({"image": image, "network": "none", "user": "65534:65534",
                        "workspace": "read-write" if writable else "read-only", "mode": snapshot.mode,
                        "memory": "256m", "cpus": "1"}, indent=2),
            encoding="utf-8",
        )

    def command(self, name):
        workspace = str(self.snapshot.workspace.resolve())
        if "," in workspace:
            raise ValueError("工作副本目录不能含逗号。")
        masks = []
        if self.snapshot.mode == "direct":
            for path, is_directory in self._protected.items():
                if any(c in path for c in (",", "\n", "\r")):
                    raise ValueError("受保护路径不能安全地挂载；请使用隔离模式。")
                source = self.snapshot.session / ("mask-dir" if is_directory else "mask-file")
                if is_directory:
                    source.mkdir(exist_ok=True)
                elif not source.exists():
                    source.write_bytes(b"")
                masks.extend(["--mount", f"type=bind,source={source},target=/workspace/{path},readonly"])
        return [
            self.docker, "create", "--pull=never", "--name", name, "-i",
            "--network=none", "--user", "65534:65534", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32",
            "--memory=256m", "--memory-swap=256m", "--cpus=1",
            *( ["--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777"] if self.writable else []),
            "--mount", f"type=bind,source={workspace},target=/workspace" + ("" if self._mount_writable else ",readonly"),
            *masks,
            "--workdir", "/workspace", self.image, "python", "-I", "-B", "-c", self.worker,
            *( ["--writable"] if self.writable else []),
        ]

    def _call(self, action, args):
        with self._lock:
            if self._cancelled.is_set():
                raise RuntimeError("会话已取消，不再执行工具。")
            return self._execute_call(action, args)

    def _execute_call(self, action, args):
        name = "cli-rb-" + uuid4().hex
        if self.snapshot.mode == "direct":
            from cli_rb.workspace.live_workspace import protected_paths
            self._protected = protected_paths(self.snapshot.workspace)
        request = json.dumps({"action": action, "args": args, "protected_paths": list(self._protected)}, ensure_ascii=False)
        journal = self.logs / f"{name}-action.json"
        journal.write_text(json.dumps({"action": action, "state": "started"}), encoding="utf-8")
        started = time.monotonic()
        phase = 'create_container'
        create_seconds = None
        try:
            # Creation does not execute the worker. Cancellation must either stop
            # creation or see its registered name before returning from cleanup.
            with self._active_lock:
                if self._cancelled.is_set():
                    raise RuntimeError("会话已取消，不再执行工具。")
                self._active.add(name)
                created = subprocess.run(self.command(name), capture_output=True, timeout=30)
                create_seconds = time.monotonic() - started
                if created.returncode:
                    raise RuntimeError("容器创建失败；没有执行工具。")
            # A late start cannot recreate a container removed by cancel_all.
            if self._cancelled.is_set():
                raise RuntimeError("会话已取消，不再执行工具。")
            phase = 'start_and_execute'
            run = subprocess.run(
                [self.docker, "start", "--attach", "--interactive", name],
                input=request, capture_output=True, text=True,
                encoding="utf-8", timeout=(args.get("timeout_seconds", 60) + 15 if action == "command" else 30),
            )
            if run.returncode:
                result = "工具错误：容器执行失败；请检查 Docker。没有回退到宿主机。"
            else:
                result = json.loads(run.stdout)["result"]
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            interrupted_seconds = time.monotonic() - started
            cleanup = subprocess.run(
                [self.docker, "rm", "--force", name], capture_output=True, timeout=15,
            )
            (self.logs / f"{name}-interrupted.json").write_text(
                json.dumps({"request": json.loads(request), "interrupted": True,
                            "phase": phase, "error_type": type(exc).__name__,
                            "timeout_seconds": getattr(exc, 'timeout', None),
                            "create_seconds": create_seconds,
                            "elapsed_before_cleanup_seconds": interrupted_seconds,
                            "worker_vs_startup_time": "unknown",
                            "cleanup_exit": cleanup.returncode}), encoding="utf-8",
            )
            raise
        finally:
            # Also remove containers that were created but never started.
            try:
                subprocess.run([self.docker, "rm", "--force", name], capture_output=True, timeout=15)
            finally:
                with self._active_lock:
                    self._active.discard(name)
        (self.logs / f"{name}.json").write_text(
            json.dumps({"request": json.loads(request), "result": result,
                        "exit_code": run.returncode}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        journal.write_text(json.dumps({"action": action,
                                       "state": "cancelled" if self._cancelled.is_set() else ("returned" if run.returncode == 0 else "uncertain"),
                                       "container_exit": run.returncode}), encoding="utf-8")
        (self.logs / f"{name}.txt").write_text(result, encoding="utf-8")
        if action == "command":
            try:
                outcome = json.loads(result)
                self.command_evidence.append({"command": args["command"],
                    "exit_code": outcome["exit_code"], "timed_out": outcome["timed_out"], "log_id": name})
                return command_feedback(result, name)
            except (ValueError, KeyError, TypeError):
                self.command_evidence.append({"command": args["command"], "status": "uncertain", "log_id": name})
        if len(result) > 16000:
            hint = f"日志 ID={name}，可用 read_tool_log 从 offset=16000 继续读取" if self.writable else "请缩小 max_lines/limit；完整结果已保存至会话日志"
            return result[:16000] + f"\n[输出截断；{hint}。]"
        return result

    def cancel_all(self):
        self._cancelled.set()
        with self._active_lock:
            names = list(self._active)
        for name in names:
            subprocess.run([self.docker, "rm", "--force", name], capture_output=True, timeout=15)

    def read_file(self, path, start_line=1, max_lines=80):
        return self._call("read", {"path": path, "start_line": start_line, "max_lines": max_lines})

    def environment_facts(self):
        return json.loads(self._call('environment', {}))

    def list_files(self, path=".", offset=0, limit=50):
        return self._call("list", {"path": path, "offset": offset, "limit": limit})

    def list_directory(self, path='.', offset=0, limit=50):
        return self._call('directory', {'path': path, 'offset': offset, 'limit': limit})

    def find_files(self, pattern, path='.', offset=0, limit=50):
        return self._call('find', {'pattern': pattern, 'path': path, 'offset': offset, 'limit': limit})

    def search_files(self, query, path=".", offset=0, limit=50):
        return self._call("search", {"query": query, "path": path, "offset": offset, "limit": limit})
