"""Tool/backend boundary. This first slice only supplies an in-memory fixture."""

from pathlib import PurePosixPath
from typing import Protocol

from langchain_core.tools import tool


class ReadBackend(Protocol):
    def read_file(self, path: str, start_line: int, max_lines: int) -> str: ...


class FixtureBackend:
    """No disk access, commands, network, or writable operations."""

    def __init__(self, files: dict[str, str]):
        self._files = dict(files)

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 80) -> str:
        parsed = PurePosixPath(path)
        if not path or parsed.is_absolute() or ".." in parsed.parts or "\\" in path or ":" in path:
            raise ValueError("只允许工作区内的相对路径。")
        if start_line < 1 or not 1 <= max_lines <= 200:
            raise ValueError("start_line 必须 >= 1；max_lines 必须在 1–200 之间。")
        key = str(parsed)
        if key not in self._files:
            raise FileNotFoundError("工作区中不存在该文件。")
        lines = self._files[key].splitlines()
        end = min(start_line - 1 + max_lines, len(lines))
        selected = lines[start_line - 1:end]
        result = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start_line))
        if end < len(lines):
            result += f"\n[更多内容：从第 {end + 1} 行继续读取]"
        return result or "[文件为空或已到末尾]"


def make_read_tool(backend: ReadBackend):
    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 80) -> str:
        """Read numbered lines of a workspace file using a relative POSIX path."""
        try:
            return backend.read_file(path, start_line, max_lines)
        except (ValueError, FileNotFoundError) as error:
            return f"工具错误：{error}"

    return read_file
