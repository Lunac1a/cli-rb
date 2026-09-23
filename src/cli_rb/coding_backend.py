"""Writable container tools and fixed-destination application tools."""

import re

from cli_rb.docker_backend import DockerReadBackend
from cli_rb.patches import PatchService


class DockerCodingBackend(DockerReadBackend):
    def __init__(self, snapshot, docker, image, approve=None):
        self.patches = PatchService(snapshot)
        self.approve = approve
        self.created_paths = set()
        super().__init__(snapshot, docker, image, writable=True)

    def _call(self, action, args, *, approved_write=False):
        with self._lock:
            direct = self.snapshot.mode == "direct"
            if direct and action == "delete" and args["path"] not in self.created_paths:
                if not self.approve or not self.approve(f"删除已有文件：{args['path']}"):
                    return "授权未获得，删除未执行。不要绕过此决定。"
            self._mount_writable = not (direct and action == "command" and not approved_write)
            try:
                result = super()._call(action, args)
                if action == "create" and "已更新" in result:
                    self.created_paths.add(args["path"])
                if action == "delete" and "已删除" in result:
                    self.created_paths.discard(args["path"])
                return result
            finally:
                self._mount_writable = self.writable

    def run_write_command(self, command, timeout_seconds=60):
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("命令超时必须为 1–60 秒。")
        with self._lock:
            if self.snapshot.mode == "direct":
                if not self.approve or not self.approve(f"允许此命令读写本地仓库（可修改或删除文件）：\n{command}"):
                    return "授权未获得，命令未执行。不要绕过此决定。"
            return self._call("command", {"command": command, "timeout_seconds": timeout_seconds}, approved_write=True)

    def create_file(self, path, content):
        return self._call("create", {"path": path, "content": content})

    def edit_file(self, path, old_text, new_text):
        return self._call("edit", {"path": path, "old_text": old_text, "new_text": new_text})

    def delete_file(self, path, expected_content):
        return self._call("delete", {"path": path, "expected_content": expected_content})

    def run_command(self, command, timeout_seconds=60):
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("命令超时必须为 1–60 秒。")
        return self._call("command", {"command": command, "timeout_seconds": timeout_seconds})

    def show_changes(self, offset=0, max_chars=12000):
        self._page_args(offset, max_chars)
        with self._lock:
            patch = self.patches.diff()
            return patch[offset:offset + max_chars] + (
                f"\n[继续读取：offset={offset + max_chars}]" if len(patch) > offset + max_chars else ""
            ) or "[没有变更或已到末尾]"

    def export_patch(self):
        with self._lock:
            if self._cancelled.is_set():
                raise RuntimeError("会话已取消。")
            return self.patches.export()

    @staticmethod
    def _page_args(offset, max_chars):
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or not 1 <= max_chars <= 16000:
            raise ValueError("offset 必须非负，max_chars 必须在 1–16000 之间。")

    def read_tool_log(self, log_id, offset=0, max_chars=12000):
        self._page_args(offset, max_chars)
        if not re.fullmatch(r"cli-rb-[0-9a-f]{32}", log_id):
            raise ValueError("无效的日志 ID。")
        text = (self.logs / f"{log_id}.txt").read_text(encoding="utf-8")
        return text[offset:offset + max_chars] + (
            f"\n[继续读取：offset={offset + max_chars}]" if len(text) > offset + max_chars else ""
        )
