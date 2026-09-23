"""Tool/backend boundary shared by the fixture and Docker workspace."""

from pathlib import PurePosixPath
import subprocess
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
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}；后端执行失败。"

    return read_file


def make_workspace_tools(backend):
    @tool
    def list_files(path: str = ".", offset: int = 0, limit: int = 50) -> str:
        """List files recursively under a workspace directory; paginate with offset and limit."""
        try:
            return backend.list_files(path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}"

    @tool
    def search_files(query: str, path: str = ".", offset: int = 0, limit: int = 50) -> str:
        """Search a literal text substring in UTF-8 workspace files; return paths and line numbers."""
        try:
            return backend.search_files(query, path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}"

    return [make_read_tool(backend), list_files, search_files]


def make_coding_tools(backend):
    def guarded(function, *args):
        try:
            return function(*args)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            return f"工具错误：{type(error).__name__}: {error if isinstance(error, ValueError) else '执行未成功，不会自动重试'}"

    @tool
    def create_file(path: str, content: str) -> str:
        """Create a new UTF-8 file in the workspace; refuses to overwrite an existing file."""
        return guarded(backend.create_file, path, content)

    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace exactly one occurrence of nonempty old_text. Read first; include enough context."""
        return guarded(backend.edit_file, path, old_text, new_text)

    @tool
    def delete_file(path: str, expected_content: str) -> str:
        """Delete a workspace file only when its full content matches expected_content from a prior read."""
        return guarded(backend.delete_file, path, expected_content)

    @tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run Linux shell command inside the offline container at /workspace. In direct mode the workspace is read-only; use run_write_command if writes are necessary. Returns exit_code, timed_out, output. Python stdlib available; missing dependencies must be reported, never install on host."""
        return guarded(backend.run_command, command, timeout_seconds)

    @tool
    def run_write_command(command: str, timeout_seconds: int = 60) -> str:
        """Request approval for an exact command with writable workspace access. Denial must be respected; do not bypass it."""
        return guarded(backend.run_write_command, command, timeout_seconds)

    @tool
    def show_changes(offset: int = 0, max_chars: int = 12000) -> str:
        """Show text patch relative to the session-start baseline; paginate with character offset."""
        return guarded(backend.show_changes, offset, max_chars)

    @tool
    def export_patch() -> str:
        """Export current text changes to an application-chosen local .patch file. Never applies to source."""
        return guarded(backend.export_patch)

    @tool
    def read_tool_log(log_id: str, offset: int = 0, max_chars: int = 12000) -> str:
        """Read saved output after truncation using the provided log ID and character offset."""
        return guarded(backend.read_tool_log, log_id, offset, max_chars)

    return make_workspace_tools(backend) + [create_file, edit_file, delete_file, run_command,
                                          run_write_command, show_changes, export_patch, read_tool_log]
