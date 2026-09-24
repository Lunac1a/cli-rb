import asyncio
import json
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest
from helpers import ScriptedModel, StreamingModel, request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from cli_rb.agent import TaskTimeLimitExceeded, astream_turn, build_agent
from cli_rb.demo import demo_backend
from cli_rb.persistence import SessionNotReady, SessionStore


@pytest.mark.parametrize('tool_request', [False, True])
def test_separate_explanation_never_binds_tools_or_changes_session(tmp_path, monkeypatch, tool_request):
    from typer.testing import CliRunner

    from cli_rb.cli import app
    from cli_rb.config import ModelConfig
    from cli_rb.snapshot import create_session

    class UnboundModel(ScriptedModel):
        def bind_tools(self, *args, **kwargs):
            pytest.fail('Explanation must not bind tools')

        def _generate(self, *args, **kwargs):
            assert kwargs.get('max_tokens') == 4096
            return super()._generate(*args, **kwargs)

    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.txt').write_text('before')
    snapshot = create_session(repo, tmp_path / 'sessions')
    (repo / 'a.txt').write_text('after')
    metadata = {'state': 'blocked', 'model_config': {'model': 'fake'}}
    (snapshot.session / 'conversation.json').write_text(json.dumps(metadata))
    logs = snapshot.session / 'tool-logs'
    logs.mkdir()
    (logs / 'edit-action.json').write_text('{"action":"edit","state":"started"}')
    (logs / 'cmd-action.json').write_text('{"action":"command","state":"returned"}')
    (logs / 'cmd.json').write_text(json.dumps({'result': '{"exit_code":1,"timed_out":false}'}))
    reply = request('a.txt', 'forbidden') if tool_request else AIMessage(content='文件有变化，未确认测试结果。')
    model = UnboundModel(replies=[reply])
    monkeypatch.setattr(ModelConfig, 'build_model', lambda self: model)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = CliRunner().invoke(app, ['explain-session', str(snapshot.session), '--request', '修改并测试'])
    assert result.exit_code == (2 if tool_request else 0), result.output
    assert ('已丢弃' if tool_request else reply.text) in result.output
    assert len(model.seen) == 1 and '-before' in model.seen[0][-1].content
    evidence = model.seen[0][-1].content.split('只读动作记录（JSON）：\n')[1].split('\n\n当前文件')[0]
    actions = {item['工具动作']: item for item in json.loads(evidence)['动作记录']}
    assert actions['command']['命令退出码'] == 1
    assert '命令退出码' not in actions['edit'] and '不确定' in actions['edit']['结果判断']
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_inspection_reports_uncertainty_without_changing_records(tmp_path):
    from typer.testing import CliRunner

    from cli_rb.cli import app

    (tmp_path / 'conversation.json').write_text('{"state": "blocked"}')
    logs = tmp_path / 'tool-logs'
    logs.mkdir()
    for name, action, state, result in [
        ('edit', 'edit', 'returned', 'edited'),
        ('test', 'command', 'returned', '{"exit_code": 1, "timed_out": false}'),
        ('pending', 'command', 'started', None),
    ]:
        (logs / f'{name}-action.json').write_text(json.dumps({'action': action, 'state': state}))
        if result is not None:
            (logs / f'{name}.json').write_text(json.dumps({'result': result}))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = CliRunner().invoke(app, ['inspect-session', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert '退出码 1' in result.output and '不代表成功' in result.output
    assert '已返回记录 2；结果不确定记录 1' in result.output
    assert '已阻止恢复' in result.output
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_inspection_treats_damaged_or_missing_evidence_as_unknown(tmp_path):
    from cli_rb.persistence import inspect_session

    (tmp_path / 'conversation.json').write_text('{"state": []}')
    logs = tmp_path / 'tool-logs'
    logs.mkdir()
    (logs / 'broken-action.json').write_text('{')
    (logs / 'missing-action.json').write_text('{"state": "returned"}')
    report = inspect_session(tmp_path)
    assert '没有有效的会话状态' in report
    assert '已返回记录 0；结果不确定记录 2' in report


@asynccontextmanager
async def opened(path, model, *, create=False, **kwargs):
    with SessionStore(path, create=create, model_config={'base_url': 'http://localhost:1234/v1',
                                                        'model': 'fixture'}) as store:
        async with AsyncSqliteSaver.from_conn_string(str(path / 'checkpoints.sqlite')) as saver:
            agent = build_agent(model, demo_backend(), checkpointer=saver, session=store, **kwargs)
            yield agent, store


async def run_turn(agent, text):
    return [event async for event in astream_turn(agent, [HumanMessage(content=text)])]


def test_reopen_keeps_history_without_duplicate_or_replayed_tools(tmp_path):
    async def first():
        model = ScriptedModel(replies=[request('README.md', 'read'), AIMessage(content='已读取')])
        async with opened(tmp_path, model, create=True) as (agent, store):
            await run_turn(agent, '记住项目代号海蓝')
            assert store.data['state'] == 'idle'
    asyncio.run(first())

    async def second():
        model = ScriptedModel(replies=[AIMessage(content='海蓝'), AIMessage(content='再见')])
        async with opened(tmp_path, model, model_call_limit=1) as (agent, store):
            history = await store.history(agent)
            assert len(history) == 4
            await run_turn(agent, '项目代号是什么？')
            assert len(model.seen) == 1
            assert sum(isinstance(m, ToolMessage) for m in model.seen[0]) == 1
            assert sum(isinstance(m, HumanMessage) for m in model.seen[0]) == 2
            await run_turn(agent, '再见')  # Per-run model budget resets after checkpoints.
            assert len(await store.history(agent)) == 8
    asyncio.run(second())


def test_tool_free_recovery_is_in_resumed_checkpoint(tmp_path):
    async def first():
        model = ScriptedModel(replies=[AIMessage(content=''), AIMessage(content='补全内容')])
        async with opened(tmp_path, model, create=True) as (agent, store):
            await run_turn(agent, '你好')
            assert (await store.history(agent))[-1].text == '补全内容'
    asyncio.run(first())

    async def second():
        async with opened(tmp_path, ScriptedModel(replies=[])) as (agent, store):
            history = await store.history(agent)
            assert len(history) == 3 and history[-1].text == '补全内容'
    asyncio.run(second())


def test_timeout_marks_blocked_and_cannot_be_resumed(tmp_path):
    async def first():
        model = StreamingModel(replies=[], block=True)
        async with opened(tmp_path, model, create=True, task_timeout_seconds=0.15) as (agent, store):
            with pytest.raises(TaskTimeLimitExceeded):
                await run_turn(agent, '继续')
            assert store.data['state'] == 'blocked'
    asyncio.run(first())
    with pytest.raises(SessionNotReady, match='未完整结束'):
        with SessionStore(tmp_path):
            pytest.fail('unsafe resume')


def test_uncertain_action_blocks_even_idle_checkpoint(tmp_path):
    async def first():
        async with opened(tmp_path, ScriptedModel(replies=[AIMessage(content='正常')]), create=True) as (agent, _):
            await run_turn(agent, '你好')
    asyncio.run(first())
    logs = tmp_path / 'tool-logs'
    logs.mkdir()
    (logs / 'x-action.json').write_text(json.dumps({'state': 'started'}))
    with pytest.raises(SessionNotReady, match='不确定'):
        with SessionStore(tmp_path):
            pass


def test_session_lock_rejects_second_owner_and_releases(tmp_path):
    with SessionStore(tmp_path, create=True):
        with pytest.raises(SessionNotReady, match='占用'):
            with SessionStore(tmp_path):
                pass
    with sqlite3.connect(tmp_path / 'conversation-lock.sqlite', timeout=0) as connection:
        connection.execute('BEGIN EXCLUSIVE')


def test_checkpoint_identity_mismatch_is_rejected(tmp_path):
    async def run():
        async with opened(tmp_path, ScriptedModel(replies=[AIMessage(content='正常')]), create=True) as (agent, store):
            await run_turn(agent, '你好')
            store.data['checkpoint_id'] = 'stale'
            with pytest.raises(SessionNotReady, match='不一致'):
                await store.history(agent)
    asyncio.run(run())


def test_process_crash_releases_lock_but_running_session_stays_blocked(tmp_path):
    code = '''
import asyncio, os, sys
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from cli_rb.persistence import SessionStore
from cli_rb.agent import build_agent
from cli_rb.demo import DemoModel, demo_backend
async def main():
    path = Path(sys.argv[1])
    with SessionStore(path, create=True, model_config={}) as store:
        async with AsyncSqliteSaver.from_conn_string(str(path / 'checkpoints.sqlite')) as saver:
            agent = build_agent(DemoModel(), demo_backend(), checkpointer=saver, session=store)
            await store.begin(agent)
            os._exit(0)
asyncio.run(main())
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True, timeout=15)
    with pytest.raises(SessionNotReady, match='未完整结束'):
        with SessionStore(tmp_path):
            pass


def test_modified_baseline_rejected(tmp_path):
    from cli_rb.snapshot import create_session, load_snapshot
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo / 'a.txt').write_text('before')
    snapshot = create_session(repo, tmp_path / 'sessions')
    (snapshot.session / 'workspace/a.txt').write_text('tampered')
    with pytest.raises(ValueError, match='基线已改变'):
        load_snapshot(snapshot.session)


def test_pending_tool_checkpoint_is_never_replayed_even_with_idle_marker(tmp_path):
    async def run():
        model = ScriptedModel(replies=[request('README.md', 'pending')])
        async with opened(tmp_path, model, create=True) as (agent, store):
            await agent.graph.ainvoke({'messages': [HumanMessage(content='读取')]},
                                     store.config, interrupt_before=['tools'])
            state = await agent.graph.aget_state(store.config)
            assert state.next == ('tools',)
            store.data.update(state='idle', checkpoint_id=state.config['configurable']['checkpoint_id'])
            store.save()
            with pytest.raises(SessionNotReady, match='待执行节点'):
                await run_turn(agent, '继续')
            assert len(model.seen) == 1
            assert not any(isinstance(m, ToolMessage) for m in state.values['messages'])
    asyncio.run(run())


def test_confirmed_recovery_keeps_old_checkpoint_and_closes_pending_call(tmp_path):
    async def run():
        model = ScriptedModel(replies=[request('README.md', 'pending'), AIMessage(content='只回答新要求')])
        async with opened(tmp_path, model, create=True) as (agent, store):
            await store.begin(agent)
            await agent.graph.ainvoke({'messages': [HumanMessage(content='旧要求')]},
                                     store.config, interrupt_before=['tools'])
            store.block()
            old_config = store.config
            state = await agent.graph.aget_state(old_config)
            assert state.next == ('tools',)
            history = await store.recover(agent, state, store.action_fingerprints())
            assert len(model.seen) == 1  # Recovery itself never calls a model.
            assert any(isinstance(m, ToolMessage) and m.status == 'error' for m in history)
            assert (await agent.graph.aget_state(old_config)).next == ('tools',)
            assert store.config != old_config and store.data['review_writes']
            await run_turn(agent, '只说明现状，不执行旧要求')
            assert len(model.seen) == 2
            assert (await store.history(agent))[-1].text == '只回答新要求'
    asyncio.run(run())
    with SessionStore(tmp_path) as reopened:
        assert '-continue-' in reopened.config['configurable']['thread_id']
        assert reopened.data['state'] == 'idle'


def test_recovery_failed_commit_or_changed_evidence_never_switches_active_state(tmp_path, monkeypatch):
    async def run():
        async with opened(tmp_path, ScriptedModel(replies=[]), create=True) as (agent, store):
            store.block()
            original = store.path.read_bytes()
            state = await agent.graph.aget_state(store.config)
            old_config = store.config

            def fail():
                raise OSError('simulated interruption before pointer commit')

            monkeypatch.setattr(store, 'save', fail)
            with pytest.raises(OSError):
                await store.recover(agent, state, {})
            assert store.path.read_bytes() == original and store.config == old_config
            logs = tmp_path / 'tool-logs'
            logs.mkdir()
            (logs / 'late-action.json').write_text('{"state":"started"}')
            with pytest.raises(SessionNotReady, match='改变'):
                await store.recover(agent, state, {})
            assert store.path.read_bytes() == original
    asyncio.run(run())


def test_cli_requires_confirmation_before_recovering_pending_work(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cli_rb import cli

    monkeypatch.setattr(cli, 'ensure_session_stopped', lambda *args: None)
    model = ScriptedModel(replies=[request('README.md', 'pending')])
    with SessionStore(tmp_path, create=True, model_config={}) as store, asyncio.Runner() as runner:
        connection = AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoints.sqlite'))
        saver = runner.run(connection.__aenter__())
        try:
            agent = build_agent(model, demo_backend(), session=store, checkpointer=saver)
            runner.run(store.begin(agent))
            runner.run(agent.graph.ainvoke({'messages': [HumanMessage(content='旧要求')]},
                                          store.config, interrupt_before=['tools']))
            store.block()
            original = store.path.read_bytes()
            backend = SimpleNamespace(patches=SimpleNamespace(diff=lambda: 'file difference'), review_writes=False)
            monkeypatch.setattr(cli.typer, 'confirm', lambda *args, **kwargs: False)
            assert cli.resume_history(runner, store, agent, backend, 'docker') is None
            assert store.path.read_bytes() == original and len(model.seen) == 1
            monkeypatch.setattr(cli.typer, 'confirm', lambda *args, **kwargs: True)
            history = cli.resume_history(runner, store, agent, backend, 'docker')
            assert history and backend.review_writes and len(model.seen) == 1
            assert not runner.run(agent.graph.aget_state(store.config)).next
        finally:
            runner.run(connection.__aexit__(None, None, None))
