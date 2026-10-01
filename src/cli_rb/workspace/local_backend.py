"""Explicitly authorized host execution; this is not a sandbox."""
import json
import hashlib
import os
import shutil
import subprocess
import threading
from uuid import uuid4

from cli_rb.workspace.coding_backend import DockerCodingBackend
from cli_rb.workspace.patches import PatchService
from cli_rb.workspace.container_worker import execute, resolve
from cli_rb.workspace.live_workspace import protected_paths
from cli_rb.quality.verification import VerificationLedger
from cli_rb.workspace.approval_policy import command_decision
from cli_rb.workspace.patches import file_bytes
from cli_rb.workspace.command_review import requires_human


def project_environment(root):
    env = os.environ.copy()
    paths = []
    for name in ('.venv', 'venv'):
        directory = root / name / ('Scripts' if os.name == 'nt' else 'bin')
        if (directory / ('python.exe' if os.name == 'nt' else 'python')).is_file():
            paths.append(str(directory))
            env['VIRTUAL_ENV'] = str(directory.parent)
            env.pop('PYTHONHOME', None)
            break
    if (root / 'node_modules/.bin').is_dir():
        paths.append(str(root / 'node_modules/.bin'))
    env['PATH'] = os.pathsep.join([*paths, env.get('PATH', '')])
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


class LocalCodingBackend(DockerCodingBackend):
    def __init__(self, snapshot, approve=None):
        if snapshot.mode != 'direct':
            raise ValueError('本地执行需要直接工作区。')
        self.snapshot, self.approve = snapshot, approve
        self.patches = PatchService(snapshot)
        self.review_writes = False
        self.created_paths = set()
        self.auto_deleted = set()
        self.reviewer = None
        self.user_task = ''
        self._lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._active = set()
        self._cancelled = threading.Event()
        self.command_evidence = []
        self.logs = snapshot.session / 'tool-logs'
        self.logs.mkdir(exist_ok=True)
        self.env = project_environment(snapshot.workspace)
        self.shell = shutil.which('powershell.exe') if os.name == 'nt' else shutil.which('bash')
        if not self.shell:
            raise ValueError('未找到本地命令解释器。')
        self.execution_rules = (
            f'Commands execute on the local host using {self.shell}, in {snapshot.workspace}. '
            f'Platform: {os.name}. Python: {shutil.which("python", path=self.env["PATH"])}; '
            f'Node: {shutil.which("node", path=self.env["PATH"])}. '
            'Project .venv/venv and node_modules/.bin are preferred on PATH. '
            'Routine simple local inspection, test and syntax-check commands are automatically approved by policy. '
            'Other commands, including pushes, destructive shell operations and installations, require human approval. '
            'Commands run with the current user privileges and are NOT sandboxed or read-only. '
            'Use host shell syntax, not Linux /workspace paths or Bash heredocs on Windows. '
            'Do not read credentials, operate outside the authorized project, install globally, '
            'change system settings, or access network services without explicit user authorization. '
            'Prefer existing project dependencies; request approval for missing project dependencies. '
            'This local mode has no automatic browser-control tool.'
        )
        (snapshot.session / 'execution.json').write_text(json.dumps(
            {'mode': 'local', 'shell': self.shell}, ensure_ascii=False), encoding='utf-8')
        self.verification = VerificationLedger(self)

    def _call(self, action, args, **kwargs):
        with self._lock:
            if self._cancelled.is_set():
                raise RuntimeError('会话已取消。')
            if action == 'command':
                return self._run_local(args)
            if action == 'delete' and args['path'] not in self.created_paths:
                relative = resolve(self.snapshot.workspace, args['path']).relative_to(self.snapshot.workspace).as_posix()
                if any(relative == p or relative.startswith(p + '/') for p in protected_paths(self.snapshot.workspace)):
                    raise ValueError('该路径受保护。')
                content = file_bytes(self.snapshot.workspace, relative)
                backed_up = (hashlib.sha256(content).hexdigest() == self.snapshot.manifest['files'].get(relative))
                automatic = backed_up and len(self.auto_deleted) < 10
                reason = '文件与会话备份一致，可恢复的少量删除' if automatic else '文件没有当前备份或删除数量较多'
                if not self._authorize(f"删除已有文件：{args['path']}", automatic, reason):
                    return '授权未获得，删除未执行。不要绕过此决定。'
                if automatic:
                    self.auto_deleted.add(args['path'])
            result = execute(self.snapshot.workspace, {'action': action, 'args': args,
                             'protected_paths': list(protected_paths(self.snapshot.workspace))}, writable=True)
            if action == 'create' and '已更新' in result:
                self.created_paths.add(args['path'])
            if action == 'delete' and '已删除' in result:
                self.created_paths.discard(args['path'])
            return self._record(action, args, result)

    def run_write_command(self, command, timeout_seconds=60):
        return self.run_command(command, timeout_seconds)

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        if os.name == 'nt':
            result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    capture_output=True, timeout=10)
            if result.returncode and process.poll() is None:
                raise RuntimeError('无法确认本地命令已停止，请检查任务管理器。')
        else:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

    def cancel_all(self):
        self._cancelled.set()
        with self._active_lock:
            for process in list(self._active):
                self._terminate(process)

    def _run_local(self, args):
        command = args['command']
        automatic, reason = command_decision(command)
        if requires_human(command):
            automatic, reason = False, '涉及重要操作，需要确认'
        elif not automatic and self.reviewer:
            automatic, reason = self.reviewer.review(command, self.user_task, self.snapshot.workspace)
        if not self._authorize(
                '允许在本机执行以下命令？命令可读写文件、访问网络，权限不限于项目目录。\n' +
                f'目录：{self.snapshot.workspace}\n{command}', automatic, reason):
            return json.dumps({'exit_code': None, 'timed_out': False, 'denied': True,
                               'output': '用户未授权，命令未执行。不得绕过。'}, ensure_ascii=False)
        name = 'cli-rb-' + uuid4().hex
        output = self.logs / f'{name}-output.txt'
        argv = ([self.shell, '-NoProfile', '-NonInteractive', '-Command',
                 "$ErrorActionPreference='Stop'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new();\n" + command + '\nif ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }']
                if os.name == 'nt' else [self.shell, '-o', 'pipefail', '-c', command])
        timed_out = False
        with output.open('wb') as stream:
            with self._active_lock:
                if self._cancelled.is_set():
                    raise RuntimeError('会话已取消，未启动命令。')
                process = subprocess.Popen(argv, cwd=self.snapshot.workspace, env=self.env,
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    start_new_session=os.name != 'nt')
                self._active.add(process)
            journal = self.logs / f'{name}-action.json'
            journal.write_text(json.dumps({'action': 'command', 'state': 'started', 'pid': process.pid}), encoding='utf-8')
            try:
                process.wait(timeout=args['timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
            finally:
                if process.poll() is None:
                    self._terminate(process)
                with self._active_lock:
                    self._active.discard(process)
        result = json.dumps({'exit_code': process.returncode, 'timed_out': timed_out,
                             'output': output.read_text(encoding='utf-8', errors='replace')}, ensure_ascii=False)
        response = self._record('command', args, result, name)
        journal.write_text(json.dumps({'action': 'command', 'state': 'cancelled' if self._cancelled.is_set() else 'returned', 'pid': process.pid}), encoding='utf-8')
        return response

    def enable_review(self, config):
        from cli_rb.workspace.command_review import CommandReview
        self.reviewer = CommandReview(config, self.snapshot.session)
        self.execution_rules += (
            ' Complex routine commands receive independent automatic review. '
            'Pushes, substantial deletion and system-level changes require human confirmation. '
            'There is no OS sandbox. Never circumvent a rejection or conceal command effects.')
        (self.snapshot.session / 'execution.json').write_text(json.dumps(
            {'mode': 'local', 'approval': 'independent-review', 'sandbox': False,
             'review_limit': 20, 'review_output_limit': 2048, 'review_timeout_seconds': 30},
            ensure_ascii=False), encoding='utf-8')

    def _authorize(self, action, automatic, reason):
        accepted = automatic or bool(self.approve and self.approve(f'{reason}\n{action}'))
        with (self.snapshot.session / 'approvals.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'action': action, 'automatic': automatic,
                                     'accepted': accepted, 'reason': reason}, ensure_ascii=False) + '\n')
        return accepted

    def _record(self, action, args, result, name=None):
        name = name or 'cli-rb-' + uuid4().hex
        (self.logs / f'{name}.json').write_text(json.dumps(
            {'request': {'action': action, 'args': args}, 'result': result},
            ensure_ascii=False), encoding='utf-8')
        (self.logs / f'{name}.txt').write_text(result, encoding='utf-8')
        if action == 'command':
            data = json.loads(result)
            self.command_evidence.append({'command': args['command'], 'exit_code': data['exit_code'],
                                         'timed_out': data['timed_out'], 'log_id': name})
        if len(result) > 16000:
            return result[:16000] + f'\n[输出截断；使用 read_tool_log，log_id={name}，offset=16000]'
        return result
