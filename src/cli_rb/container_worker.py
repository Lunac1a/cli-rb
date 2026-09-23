"""Standalone stdlib worker, executed only by Python inside the container."""

import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile


def text_file(path):
    if not path.is_file():
        raise FileNotFoundError("文件不存在。")
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or b"\0" in data:
        raise ValueError("文件过大或属于二进制文件。")
    return data.decode("utf-8")


def execute_command(root, args):
    import resource  # Linux container only.

    command = args["command"]
    if not isinstance(command, str) or not 1 <= len(command) <= 8000:
        raise ValueError("命令长度必须在 1–8000 字符之间。")
    timeout = integer(args.get("timeout_seconds", 60), 1, 60)
    cwd = resolve(root, args.get("path", "."))
    if not cwd.is_dir():
        raise ValueError("命令工作目录不存在。")

    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))

    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            ["/bin/sh", "-c", command], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
                 "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also kill background descendants before returning the tool result.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        captured = output.read().decode("utf-8", errors="replace")
    return json.dumps({"exit_code": process.returncode, "timed_out": timed_out,
                       "output": captured, "output_limit_bytes": 8 * 1024 * 1024}, ensure_ascii=False)


def resolve(root, value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("只允许工作区内的相对路径。")
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("不允许符号链接。")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("路径越界。")
    return current


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"参数必须是 {low}–{high} 之间的整数。")
    return value


def execute(root, request, *, writable=False):
    action = request["action"]
    args = request["args"]
    path = resolve(root, args.get("path", "."))
    relative = path.relative_to(root).as_posix()
    protected = request.get("protected_paths", [])
    if any(relative == p or relative.startswith(p + "/") for p in protected):
        raise ValueError("该路径受保护，工具不能访问。")
    if action in {"create", "edit", "delete", "command"}:
        if not writable:
            raise ValueError("当前后端为只读模式。")
        if action == "command":
            return execute_command(root, args)
        if path == root or any(part in {".git", ".cli-rb"} for part in path.relative_to(root).parts):
            raise ValueError("不允许修改工作区根目录或应用元数据。")
        if action == "delete":
            if text_file(path).replace("\r\n", "\n") != args["expected_content"].replace("\r\n", "\n"):
                raise ValueError("文件内容与预期不符，请重新读取。")
            path.unlink()
            return "已删除工作区文件。"
        if action == "create":
            content = args["content"]
            if path.exists():
                raise ValueError("文件已存在，请使用 edit_file。")
        else:
            content = text_file(path)
            old = args["old_text"]
            new = args["new_text"]
            if old and content.count(old) == 0 and "\r\n" in content:
                # Model reads numbered logical lines; preserve Windows line endings on edit.
                old = old.replace("\r\n", "\n").replace("\n", "\r\n")
                new = new.replace("\r\n", "\n").replace("\n", "\r\n")
            if not old or content.count(old) != 1:
                raise ValueError("old_text 必须非空且只匹配一处，请重新读取并提供足够上下文。")
            content = content.replace(old, new, 1)
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1024 * 1024 or "\0" in content:
            raise ValueError("仅支持不超过 1 MiB 的 UTF-8 文本。")
        path.parent.mkdir(parents=True, exist_ok=True)
        # A same-directory replace prevents partial edits; no model-selected host path.
        if action == "create":
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
        else:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                stream.write(content.encode("utf-8"))
                temporary = Path(stream.name)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        return "已更新工作区文件；尚未验证。"
    if action == "read":
        start = integer(args.get("start_line", 1), 1, 1000000)
        count = integer(args.get("max_lines", 80), 1, 200)
        lines = text_file(path).splitlines()
        end = min(start - 1 + count, len(lines))
        result = "\n".join(
            f"{i}: {line[:2000]}" + (" [该行超过 2000 字符，已截断]" if len(line) > 2000 else "")
            for i, line in enumerate(lines[start - 1:end], start)
        )
        if end < len(lines):
            result += f"\n[更多内容：从第 {end + 1} 行继续读取]"
        return result or "[文件为空或已到末尾]"
    if not path.is_dir():
        raise NotADirectoryError("目录不存在。")
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
                   and not any(p.relative_to(root).as_posix() == blocked or
                               p.relative_to(root).as_posix().startswith(blocked + "/") for blocked in protected))
    offset = integer(args.get("offset", 0), 0, 1000000)
    limit = integer(args.get("limit", 50), 1, 100)
    if action == "list":
        rows = [p.relative_to(root).as_posix() for p in files]
    elif action == "search":
        query = args["query"]
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise ValueError("搜索词必须是 1–200 字符的非空文本。")
        rows = []
        for file in files:
            try:
                lines = text_file(resolve(root, file.relative_to(root).as_posix())).splitlines()
            except (OSError, ValueError):
                continue
            for i, line in enumerate(lines, 1):
                if query in line:
                    rows.append(f"{file.relative_to(root).as_posix()}:{i}: {line[:500]}")
                    if len(rows) > offset + limit:
                        break
            if len(rows) > offset + limit:
                break
    else:
        raise ValueError("不支持的操作。")
    result = "\n".join(rows[offset:offset + limit]) or "[没有结果]"
    if len(rows) > offset + limit:
        result += f"\n[更多结果：offset={offset + limit}]"
    return result


if __name__ == "__main__":
    try:
        result = execute(Path("/workspace"), json.load(sys.stdin), writable="--writable" in sys.argv)
    except (ValueError, OSError, KeyError, TypeError) as error:
        result = f"工具错误：{type(error).__name__}: {error}"
    print(json.dumps({"result": result}, ensure_ascii=False))
