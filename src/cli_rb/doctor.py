"""Bounded, read-only checks; reachability is not a model acceptance test."""

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

import httpx

from cli_rb.config import ModelConfig
from cli_rb.docker_backend import docker_executable


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def command_check(name: str, args: list[str]) -> Check:
    executable = shutil.which(args[0])
    if not executable:
        return Check(name, False, "未找到可执行程序")
    try:
        result = subprocess.run(
            [executable, *args[1:]], capture_output=True, timeout=10, check=False,
        )
        output = result.stdout + result.stderr
        encoding = "utf-16-le" if b"\x00" in output else "utf-8"
        detail = output.decode(encoding, errors="replace").strip()[:1500]
        return Check(name, result.returncode == 0, detail or f"exit={result.returncode}")
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(name, False, type(error).__name__)


def inspect_environment(config: ModelConfig) -> list[dict]:
    checks = [Check("Python", platform.python_version_tuple()[:2] == ("3", "12"),
                    platform.python_version())]
    checks.append(command_check("uv", ["uv", "--version"]))
    if platform.system() == "Windows":
        checks.append(command_check("WSL", ["wsl", "--status"]))
    try:
        checks.append(command_check("Docker", [docker_executable(), "version"]))
    except RuntimeError as error:
        checks.append(Check("Docker", False, str(error)))
    try:
        response = httpx.get(
            config.base_url.rstrip("/") + "/models", timeout=5, trust_env=False,
            headers={"Authorization": f"Bearer {os.getenv('CLI_RB_API_KEY', 'lm-studio')}"},
        )
        response.raise_for_status()
        models = [item["id"] for item in response.json()["data"]]
        ok = bool(config.model) and config.model in models
        selection = "已选择模型" if ok else "尚未配置可用的 CLI_RB_MODEL"
        checks.append(Check("模型服务", ok,
                            f"服务可达；模型 ID：{models}；{selection}；尚未验证工具调用或显存"))
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
        checks.append(Check("模型服务", False, type(error).__name__))
    return [asdict(check) for check in checks]
