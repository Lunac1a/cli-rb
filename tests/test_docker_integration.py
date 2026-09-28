"""Opt-in real Docker checks; unavailable Docker is a failure when requested."""

import json
import os
import subprocess
import time
from uuid import uuid4

import pytest

from cli_rb.workspace.coding_backend import DockerCodingBackend
from cli_rb.workspace.docker_backend import DockerReadBackend, prepare_image
from cli_rb.workspace.snapshot import create_snapshot

pytestmark = pytest.mark.skipif(
    os.getenv("CLI_RB_DOCKER_TESTS") != "1", reason="set CLI_RB_DOCKER_TESTS=1 for real Docker",
)


@pytest.mark.parametrize('pause_at', ['scan', 'create', 'start'])
def test_cancel_before_start_never_writes_after_return(tmp_path, monkeypatch, pause_at):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from cli_rb.workspace import live_workspace
    from cli_rb.workspace.snapshot import create_session

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    target = repo / 'a.txt'
    target.write_text('before')
    backend = DockerCodingBackend(create_session(repo, tmp_path / 'sessions'), *prepare_image())
    entered, release = threading.Event(), threading.Event()
    original_scan, original_run = live_workspace.protected_paths, subprocess.run

    def pause():
        entered.set()
        assert release.wait(10)

    def scan(*args):
        result = original_scan(*args)
        if pause_at == 'scan':
            pause()
        return result

    def run(args, **kwargs):
        if args[0] == backend.docker and args[1] == pause_at:
            pause()
        return original_run(args, **kwargs)

    monkeypatch.setattr(live_workspace, 'protected_paths', scan)
    monkeypatch.setattr('cli_rb.workspace.docker_backend.subprocess.run', run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(backend.edit_file, 'a.txt', 'before', 'after')
        try:
            assert entered.wait(10)
            cancellation = pool.submit(backend.cancel_all)
            assert backend._cancelled.wait(5)
            if pause_at == 'create':
                # Cancellation must await the create/registration critical section.
                assert not cancellation.done()
                release.set()
            cancellation.result(timeout=10)
            assert target.read_text() == 'before'
        finally:
            release.set()
        try:
            result = work.result(timeout=15)
            assert '工具错误' in result
        except RuntimeError as error:
            assert '取消' in str(error)
    assert target.read_text() == 'before'
    assert not backend._active
    names = {p.name.removesuffix('-action.json') for p in backend.logs.glob('*-action.json')}
    remaining = original_run([backend.docker, 'ps', '-a', '--format', '{{.Names}}'],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    assert not names.intersection(remaining)


@pytest.mark.parametrize('mode', ['direct', 'isolated'])
def test_interrupted_command_continues_without_replay_after_review(tmp_path, mode):
    import asyncio

    from helpers import ScriptedModel, request
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from cli_rb.runtime.agent import astream_turn, build_agent
    from cli_rb.workspace.docker_backend import ensure_session_stopped
    from cli_rb.runtime.persistence import SessionStore
    from cli_rb.workspace.snapshot import create_session, load_snapshot

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.txt').write_text('baseline')
    snapshot = create_session(repo, tmp_path / 'sessions', mode)
    docker, image = prepare_image()
    first_backend = DockerCodingBackend(snapshot, docker, image, approve=lambda _: True)

    async def interrupt():
        model = ScriptedModel(replies=[AIMessage(content='', tool_calls=[{
            'name': 'run_write_command', 'args': {'command': 'echo once >> progress.txt; sleep 20'},
            'id': 'old-command', 'type': 'tool_call',
        }])])
        with SessionStore(snapshot.session, create=True, model_config={}) as store:
            async with AsyncSqliteSaver.from_conn_string(str(snapshot.session / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, first_backend, coding=True, session=store, checkpointer=saver)

                async def run():
                    async for _ in astream_turn(agent, [HumanMessage(content='写入后等待')]):
                        pass

                task = asyncio.create_task(run())
                try:
                    async with asyncio.timeout(10):
                        while not (snapshot.workspace / 'progress.txt').exists():
                            await asyncio.sleep(0.05)
                    with pytest.raises(RuntimeError, match='仍有容器'):
                        ensure_session_stopped(snapshot.session, docker)
                finally:
                    first_backend.cancel_all()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert store.data['state'] == 'blocked' and len(model.seen) == 1
    asyncio.run(interrupt())

    async def continue_new_request():
        loaded = load_snapshot(snapshot.session)
        backend = DockerCodingBackend(loaded, docker, image, approve=lambda _: False, review_writes=True)
        model = ScriptedModel(replies=[request('progress.txt', 'fresh-read'), AIMessage(content='', tool_calls=[{
            'name': 'create_file', 'args': {'path': 'done.txt', 'content': 'new authorized work'},
            'id': 'fresh-write', 'type': 'tool_call',
        }]), AIMessage(content='已读取现状并完成新的写入，没有重跑旧命令'),
            AIMessage(content='修改已保留，未运行验证；没有重跑旧命令。')])
        with SessionStore(snapshot.session, allow_interrupted=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(snapshot.session / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, coding=True, session=store, checkpointer=saver)
                ensure_session_stopped(snapshot.session, docker)
                fingerprints = store.action_fingerprints()
                state = await agent.graph.aget_state(store.config)
                await store.recover(agent, state, fingerprints)
                assert store.action_fingerprints() == fingerprints and not model.seen
                assert '未获授权' in backend.create_file('denied.txt', 'must not write')
                assert not (loaded.workspace / 'denied.txt').exists()
                approvals = []
                backend.approve = lambda action: approvals.append(action) or True
                async for _ in astream_turn(agent, [HumanMessage(content='读取现状，然后创建done.txt，不重跑旧命令')]):
                    pass
                assert approvals and store.data['state'] == 'idle'
                assert len(model.seen) == 4
                assert '程序收尾核对' in model.seen[-1][-1].text
                assert (loaded.workspace / 'progress.txt').read_text().strip() == 'once'
                assert (loaded.workspace / 'done.txt').read_text() == 'new authorized work'
                assert backend.patches.baseline.joinpath('a.txt').read_text() == 'baseline'
                # Late changes to reviewed records must block ordinary resume again.
                old_log = loaded.session / 'tool-logs' / next(iter(fingerprints))
                old_log.write_text('{}')
                with pytest.raises(RuntimeError, match='记录发生变化'):
                    await store.history(agent)
    asyncio.run(continue_new_request())
    assert (repo / 'progress.txt').exists() == (mode == 'direct')


@pytest.mark.parametrize('mode', ['direct', 'isolated'])
def test_resume_docker_edit_keeps_baseline_and_never_replays(tmp_path, mode):
    import asyncio

    from helpers import ScriptedModel, request
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from cli_rb.runtime.agent import astream_turn, build_agent
    from cli_rb.runtime.persistence import SessionStore
    from cli_rb.workspace.snapshot import create_session, load_snapshot

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.txt').write_text('before')
    snapshot = create_session(repo, tmp_path / 'sessions', mode)
    docker, image = prepare_image()

    async def run(first):
        loaded = snapshot if first else load_snapshot(snapshot.session)
        backend = DockerCodingBackend(loaded, docker, image)
        model = ScriptedModel(replies=([
            request('a.txt', 'read'), AIMessage(content='', tool_calls=[{
                'name': 'edit_file', 'args': {'path': 'a.txt', 'old_text': 'before', 'new_text': 'after'},
                'id': 'edit', 'type': 'tool_call',
            }]), AIMessage(content='已修改'), AIMessage(content='修改已保留，尚未验证。'),
        ] if first else [AIMessage(content='已有修改保留')]))
        with SessionStore(snapshot.session, create=first, model_config={}) as store:
            async with AsyncSqliteSaver.from_conn_string(str(snapshot.session / 'checkpoints.sqlite')) as saver:
                agent = build_agent(model, backend, coding=True, session=store, checkpointer=saver)
                if not first:
                    history = await store.history(agent)
                    assert len(history) == 8
                    assert '验证证据未通过核对' in history[-1].text
                async for _ in astream_turn(agent, [HumanMessage(content='修改' if first else '说明上次结果')]):
                    pass
                if not first:
                    assert len(model.seen) == 1
                    assert sum(isinstance(m, ToolMessage) for m in model.seen[0]) == 2
        assert len(list(backend.logs.glob('*-action.json'))) == 2
        assert backend.patches.baseline.joinpath('a.txt').read_text() == 'before'
        assert loaded.workspace.joinpath('a.txt').read_text() == 'after'

    asyncio.run(run(True))
    asyncio.run(run(False))
    assert (repo / 'a.txt').read_text() == ('after' if mode == 'direct' else 'before')


def test_task_deadline_stops_command_and_preserves_written_file(tmp_path):
    from helpers import ScriptedModel
    from langchain_core.messages import AIMessage, HumanMessage

    from cli_rb.runtime.agent import TaskTimeLimitExceeded, build_agent, stream_turn

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'readme').write_text('fixture')
    snapshot = create_snapshot(repo, tmp_path / 'sessions')
    backend = DockerCodingBackend(snapshot, *prepare_image())
    model = ScriptedModel(replies=[AIMessage(content='', tool_calls=[{
        'name': 'run_command', 'args': {'command': 'echo saved > progress.txt; sleep 20'},
        'id': 'command', 'type': 'tool_call',
    }])])
    start = time.monotonic()
    with pytest.raises(TaskTimeLimitExceeded):
        list(stream_turn(build_agent(model, backend, coding=True, task_timeout_seconds=5),
                         [HumanMessage(content='执行')]))
    assert time.monotonic() - start < 15
    assert (snapshot.workspace / 'progress.txt').read_text().strip() == 'saved'
    assert not backend._active
    assert len(model.seen) == 1
    with pytest.raises(RuntimeError, match='取消'):
        backend.run_command('echo replayed >> progress.txt')


def test_deadline_while_waiting_for_permission_never_deletes(tmp_path):
    import asyncio

    from helpers import ScriptedModel
    from langchain_core.messages import AIMessage, HumanMessage

    from cli_rb.runtime.agent import TaskTimeLimitExceeded, build_agent
    from cli_rb.interface.cli import display_turn
    from cli_rb.workspace.snapshot import create_session

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'keep.txt').write_text('keep')
    snapshot = create_session(repo, tmp_path / 'sessions')
    entered, closed = [], []

    async def ask(action):
        entered.append(action)
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    backend = DockerCodingBackend(snapshot, *prepare_image(), approve=ask)
    model = ScriptedModel(replies=[AIMessage(content='', tool_calls=[{
        'name': 'delete_file', 'args': {'path': 'keep.txt', 'expected_content': 'keep'},
        'id': 'delete', 'type': 'tool_call',
    }])])
    agent = build_agent(model, backend, coding=True, task_timeout_seconds=0.3)
    with pytest.raises(TaskTimeLimitExceeded):
        asyncio.run(display_turn(agent, [HumanMessage(content='删除')], backend, []))
    assert entered and closed == [True]
    assert (repo / 'keep.txt').read_text() == 'keep'
    assert not list(backend.logs.glob('*-action.json'))
    assert len(model.seen) == 1


def test_tool_budget_preserves_prior_direct_edit_and_blocks_pending_batch(tmp_path):
    from helpers import ScriptedModel, request
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
    from langchain_core.messages import AIMessage, HumanMessage

    from cli_rb.runtime.agent import build_agent, stream_turn
    from cli_rb.workspace.snapshot import create_session

    repo = tmp_path / 'direct'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.txt').write_text('before')
    backend = DockerCodingBackend(create_session(repo, tmp_path / 'sessions'), *prepare_image())
    model = ScriptedModel(replies=[request('a.txt', 'read'), AIMessage(content='', tool_calls=[{
        'name': 'edit_file', 'args': {'path': 'a.txt', 'old_text': 'before', 'new_text': 'after'},
        'id': 'edit', 'type': 'tool_call',
    }]), AIMessage(content='', tool_calls=[{
        'name': 'create_file', 'args': {'path': name, 'content': 'must not execute'},
        'id': name, 'type': 'tool_call',
    } for name in ['one.txt', 'two.txt']])])
    with pytest.raises(ToolCallLimitExceededError):
        list(stream_turn(build_agent(model, backend, coding=True, tool_call_limit=3),
                         [HumanMessage(content='修改')]))
    assert (repo / 'a.txt').read_text() == 'after'
    assert not (repo / 'one.txt').exists() and not (repo / 'two.txt').exists()
    assert len(list(backend.logs.glob('*-action.json'))) == 2
    assert len(model.seen) == 3


def test_real_docker_read_only_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=excluded", encoding="utf-8")
    docker, image = prepare_image()
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    backend = DockerReadBackend(snapshot, docker, image)
    assert "app.py" in backend.list_files()
    assert "app.py:1:" in backend.search_files("greet")
    assert "return 'hello'" in backend.read_file("app.py")
    assert backend.read_file("../secret").startswith("工具错误")
    assert backend.read_file(".env").startswith("工具错误")
    # Direct OS-level probes using exactly the same run restrictions as the backend.
    name = "cli-rb-test-" + uuid4().hex
    command = backend.command(name)
    command[-1] = '''import json, os, socket
from pathlib import Path
checks = {"uid": os.getuid(), "no_docker_socket": not Path("/var/run/docker.sock").exists()}
try:
    Path("/workspace/app.py").write_text("changed")
    checks["write_denied"] = False
except OSError:
    checks["write_denied"] = True
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2).close()
    checks["network_denied"] = False
except OSError:
    checks["network_denied"] = True
status = Path("/proc/self/status").read_text()
checks["no_new_privileges"] = "NoNewPrivs:\\t1" in status
checks["no_capabilities"] = "CapEff:\\t0000000000000000" in status
print(json.dumps(checks))
'''
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        result = subprocess.run([docker, 'start', '--attach', name],
                                capture_output=True, text=True, timeout=30, check=True)
    finally:
        subprocess.run([docker, 'rm', '--force', name], capture_output=True, timeout=15)
    checks = json.loads(result.stdout)
    assert checks.pop("uid") == 65534
    assert all(checks.values()), checks
    assert (repo / "app.py").read_text() == "def greet():\n    return 'hello'\n"


def test_real_coding_edit_command_timeout_patch_and_boundary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    original = "def add(a, b):\n    return a - b\n"
    (repo / "app.py").write_text(original, encoding="utf-8")
    docker, image = prepare_image()
    snapshot = create_snapshot(repo, tmp_path / "sessions")
    backend = DockerCodingBackend(snapshot, docker, image)
    failure = json.loads(backend.run_command('python -c "from app import add; assert add(2, 3) == 5"'))
    assert failure["exit_code"] != 0 and not failure["timed_out"]
    assert "已更新" in backend.edit_file("app.py", "return a - b", "return a + b")
    passed = json.loads(backend.run_command('python -c "from app import add; assert add(2, 3) == 5; print(123)"'))
    assert passed["exit_code"] == 0 and "123" in passed["output"]
    assert "已更新" in backend.create_file("new.txt", "new\n")
    assert "已删除" in backend.delete_file("new.txt", "new\n")
    assert "工具错误" in backend.create_file("../escape", "bad")
    timeout = json.loads(backend.run_command("sleep 10", timeout_seconds=1))
    assert timeout["timed_out"] and timeout["exit_code"] != 0
    assert "未应用" in backend.export_patch()
    assert (repo / "app.py").read_text() == original
    assert "return a + b" in backend.show_changes()
    # Shell cannot write outside mounted copy or find the protected baseline.
    denied = json.loads(backend.run_command("touch /cannot-write; test -e /baseline"))
    assert denied["exit_code"] != 0 and "Read-only file system" in denied["output"]
    # A command-created symlink must not be followed by host-side patch export.
    backend.run_command("ln -s /etc/passwd link")
    with pytest.raises(ValueError, match="链接"):
        backend.export_patch()


def test_direct_workspace_edits_permissions_and_masks(tmp_path):
    from cli_rb.workspace.snapshot import create_session
    repo = tmp_path / 'live'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'app.py').write_text('value = 1\n', encoding='utf-8')
    (repo / '.env').write_text('PRIVATE_VALUE=keep', encoding='utf-8')
    (repo / '.gitignore').write_text('private/\n', encoding='utf-8')
    (repo / 'private').mkdir()
    (repo / 'private/token').write_text('keep', encoding='utf-8')
    git_head = (repo / '.git/HEAD').read_bytes()
    snapshot = create_session(repo, tmp_path / 'sessions')
    backend = DockerCodingBackend(snapshot, *prepare_image())
    assert snapshot.workspace == repo.resolve()
    assert '已更新' in backend.edit_file('app.py', 'value = 1', 'value = 2')
    assert (repo / 'app.py').read_text() == 'value = 2\n'
    assert (snapshot.session / 'workspace/app.py').read_text() == 'value = 1\n'
    (repo / 'app.py').write_text('value = 3\n', encoding='utf-8')
    assert 'value = 3' in backend.read_file('app.py')
    assert '+value = 3' in backend.show_changes()
    result = json.loads(backend.run_command('echo bad > app.py'))
    assert result['exit_code'] != 0
    assert '授权未获得' in backend.run_write_command('echo bad > app.py')
    assert '授权未获得' in backend.delete_file('app.py', 'value = 3\n')
    assert '工具错误' in backend.read_file('.env')
    masks = json.loads(backend.run_command('test ! -s .env && test ! -e .git/HEAD && test ! -e private/token'))
    assert masks['exit_code'] == 0, masks
    assert (repo / '.env').read_text() == 'PRIVATE_VALUE=keep'
    assert (repo / '.git/HEAD').read_bytes() == git_head
    approvals = []
    backend.approve = lambda action: approvals.append(action) or True
    result = json.loads(backend.run_write_command('echo approved > result.txt'))
    assert result['exit_code'] == 0, result
    assert (repo / 'result.txt').read_text().strip() == 'approved'
    assert len(approvals) == 1
    assert '已更新' in backend.create_file('new.txt', 'new')
    assert '已删除' in backend.delete_file('new.txt', 'new')
    assert len(approvals) == 1
    assert '已删除' in backend.delete_file('app.py', 'value = 3\n')
    assert len(approvals) == 2
    assert '修改已在本地仓库' in backend.export_patch()


def test_async_agent_cancellation_stops_command_before_next_model_call(tmp_path):
    import asyncio

    from helpers import ScriptedModel
    from langchain_core.messages import AIMessage, HumanMessage

    from cli_rb.runtime.agent import build_agent
    from cli_rb.interface.cli import display_turn

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'readme').write_text('fixture')
    snapshot = create_snapshot(repo, tmp_path / 'sessions')
    backend = DockerCodingBackend(snapshot, *prepare_image())
    model = ScriptedModel(replies=[AIMessage(content='', tool_calls=[{
        'name': 'run_command', 'args': {'command': 'echo saved > progress.txt; sleep 20'},
        'id': 'command-1', 'type': 'tool_call',
    }])])

    async def check():
        completed = []
        task = asyncio.create_task(display_turn(build_agent(model, backend, coding=True),
                                   [HumanMessage(content='执行命令')], backend, completed))
        try:
            async with asyncio.timeout(10):
                while not (snapshot.workspace / 'progress.txt').exists():
                    await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        assert len(completed) == 1 and completed[0].tool_calls

    asyncio.run(check())
    assert len(model.seen) == 1
    assert (snapshot.workspace / 'progress.txt').read_text().strip() == 'saved'
    assert not backend._active
    with pytest.raises(RuntimeError, match='取消'):
        backend.run_command('echo replayed >> progress.txt')
    assert (snapshot.workspace / 'progress.txt').read_text().strip() == 'saved'
