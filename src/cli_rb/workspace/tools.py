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
        """Read a workspace-relative file path (e.g. src/app.py, never /workspace/src/app.py). start_line >= 1, max_lines 1-200. Large pages stop early; follow the returned next line."""
        try:
            return backend.read_file(path, start_line, max_lines)
        except (ValueError, FileNotFoundError) as error:
            return f"工具错误：{error}"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}；后端执行失败。"

    return read_file


def make_workspace_tools(backend):
    @tool
    def list_directory(path: str = '.', offset: int = 0, limit: int = 50) -> str:
        """List one directory level without walking the repository. Directories end in /. Start here for repository navigation."""
        try:
            return backend.list_directory(path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f'工具错误：{type(error).__name__}: {error}'

    @tool
    def find_files(pattern: str, path: str = '.', offset: int = 0, limit: int = 50) -> str:
        """Find workspace-relative file paths with glob syntax (e.g. '*test*parser*.py'), without reading file contents. Paginate with offset."""
        try:
            return backend.find_files(pattern, path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f'工具错误：{type(error).__name__}: {error}'
    @tool
    def list_files(path: str = ".", offset: int = 0, limit: int = 50) -> str:
        """List files under a relative directory, e.g. '.' or 'src', never an absolute path. Paginate with offset and limit."""
        try:
            return backend.list_files(path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}"

    @tool
    def search_files(query: str, path: str = ".", offset: int = 0, limit: int = 50) -> str:
        """Search a literal substring, NOT regex. path must be a workspace-relative DIRECTORY ('.' or 'src'), not a file or absolute path. Returns file paths and line numbers."""
        try:
            return backend.search_files(query, path, offset, limit)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return f"工具错误：{type(error).__name__}"

    return [make_read_tool(backend), list_files, search_files,
            *([list_directory, find_files] if hasattr(backend, 'list_directory') else [])]


def make_coding_tools(backend):
    def guarded(function, *args):
        try:
            return function(*args)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            return f"工具错误：{type(error).__name__}: {error if isinstance(error, ValueError) else '执行未成功，不会自动重试'}"

    @tool
    def create_file(path: str, content: str) -> str:
        """Create a new UTF-8 file using a workspace-relative path (e.g. test_bug.py). No absolute paths; refuses to overwrite an existing file."""
        return guarded(backend.create_file, path, content)

    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Edit a workspace-relative file path. Replace exactly one occurrence of nonempty old_text. Read first; include enough context."""
        return guarded(backend.edit_file, path, old_text, new_text)

    @tool
    def delete_file(path: str, expected_content: str) -> str:
        """Delete a workspace-relative file only when its full content matches expected_content from a prior read."""
        return guarded(backend.delete_file, path, expected_content)

    @tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run Bash with pipefail in the offline container at /workspace; timeout_seconds 1-60. In direct mode workspace is read-only; use run_write_command for writes. Returns exit_code, timed_out, output. Nonzero includes pipeline failures; zero is NOT proof tests passed (check output and collected tests). Use read_tool_log instead of head/tail. Never install on host."""
        return guarded(backend.run_command, command, timeout_seconds)

    @tool
    def run_write_command(command: str, timeout_seconds: int = 60) -> str:
        """Run Bash with pipefail and writable workspace access; timeout_seconds 1-60. Requests approval in direct mode. Denial must be respected. Check both exit_code and output; zero alone does not prove tests passed."""
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

    @tool
    def run_verification(command: str, scope: str, timeout_seconds: int = 60, kind: str = 'test',
                         behavior_ids: list[str] | None = None, assertion: str = '',
                         purpose: str = 'verify', expected_failure: str = '') -> str:
        """Execute once with code-version evidence. Optionally link registered IDs via behavior_ids for whole-run coverage; prefer bind_verification after execution for specific nodes and describe the actual checked assertion, not just a test command. purpose=verify/regression checks current behavior; reproduce requires expected_failure text specified before execution and records a pre-fix failure. A later successful check of the same declared behavior links the reproduction without erasing history. Simple pytest commands automatically produce JUnit XML in container /tmp, preserving test selection and logs; complex commands fall back to text, unknown is NOT passing. kind=check records command success only for assertion scripts/build/lint. Read-only workspace and approval rules are unchanged. Reports cannot prove semantic coverage; inspect assertions before claiming a behavior is checked."""
        return guarded(backend.run_verification, command, scope, timeout_seconds, kind,
                       behavior_ids, assertion, purpose, expected_failure)

    @tool
    def register_behavior(behavior_id: str, expected: str, assertion: str, affected_paths: list[str]) -> str:
        """Register or revise ONE observable product behavior, independently of write_todos. expected is the observable result; assertion distinguishes correct from incorrect; affected_paths are investigated code paths (may initially be empty). Do not register investigation/edit/test-running steps. Call only for new or genuinely changed requirements, not progress. Revising a definition invalidates old bindings; existing current test evidence can be rebound if its actual assertions still establish the revised behavior."""
        return guarded(backend.verification.register_behavior, behavior_id, expected, assertion, affected_paths)

    @tool
    def bind_verification(behavior_id: str, evidence_id: str, test_nodes: list[str], assertion: str) -> str:
        """Bind existing run_verification evidence WITHOUT rerunning. Copy exact relevant node IDs from the returned test_evidence.nodes, including parameter cases needed for this behavior; assertion explains their actual checks. Only selected nodes must pass; unrelated skipped nodes do not invalidate this behavior. Failed tests elsewhere remain reported. For kind=check use test_nodes=[]; this proves command success only. Never guess nodes or narrow the selection to hide a failure relevant to the behavior."""
        return guarded(backend.verification.bind_verification, behavior_id, evidence_id, test_nodes, assertion)

    @tool
    def review_work() -> str:
        """Inspect changed paths, Python syntax/test-definition risks, and stale/failed/missing verification evidence. Read-only: never runs tests or approves semantic correctness. Legitimate test renames may produce review warnings."""
        return guarded(backend.review_work)

    return make_workspace_tools(backend) + [create_file, edit_file, delete_file, run_command,
                                          run_write_command, show_changes, export_patch, read_tool_log,
                                          *([register_behavior, run_verification, bind_verification, review_work] if hasattr(backend, 'verification') else [])]
