"""Terminal input and rendering, separate from agent and workspace execution."""

import json
import asyncio
import inspect
import os
import subprocess
import sqlite3
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.syntax import Syntax
from rich.console import Console
from rich.markdown import Markdown
from cli_rb.interface.input import chat_prompt
from cli_rb.interface.presentation import activity, print_answer, print_details, TurnProgress, progress_entries, SessionMetrics
from cli_rb.interface.workspace_selection import select_workspace
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from cli_rb.runtime.agent import (
    build_agent, stream_turn, astream_turn, TextDelta, ContextUpdate, has_reply, TaskTimeLimitExceeded,
)
from cli_rb.interface.approval import TerminalApproval, approval_dialog
from cli_rb.runtime.context import ContextLimitExceeded
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from cli_rb.config import ModelConfig
from cli_rb.interface.demo import DEMO_QUESTION, DemoModel, demo_backend
from cli_rb.interface.doctor import inspect_environment
from cli_rb.workspace.docker_backend import DockerReadBackend, prepare_image, ensure_session_stopped
from cli_rb.workspace.snapshot import create_snapshot, create_session, load_snapshot
from cli_rb.runtime.persistence import SessionStore, SessionNotReady, inspect_session, recovery_messages
from cli_rb.workspace.coding_backend import DockerCodingBackend
from cli_rb.workspace.local_backend import LocalCodingBackend

app = typer.Typer(invoke_without_command=True, help="CLI Rb：本地工作区编码助手。")
console = Console()
TaskMinutes = Annotated[int, typer.Option(
    "--task-minutes", min=1, help="每条用户请求的总时长上限（分钟），含等待授权。",
)]


def disable_remote_tracing():
    # Local by default, even when the parent shell has tracing enabled.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"


def render(message, *, show_text=True):
    if isinstance(message, ToolMessage):
        if message.status == 'error' or str(message.content).startswith('工具错误'):
            console.print('操作未成功，正在检查原因…', style='yellow')
    elif isinstance(message, AIMessage):
        for call in message.tool_calls:
            console.print(activity(call), style='dim', markup=False)
        if show_text and not message.tool_calls and message.text:
            print_answer(console, message.text, message.response_metadata.get('completion_check'))


def report_empty_reply(messages, backend, evidence_start=0):
    if messages and has_reply(messages[-1]):
        return False
    console.print("模型未返回有效的最终文字答复；以下为程序记录，不代表模型已确认需求全部完成。")
    evidence = backend.command_evidence[evidence_start:]
    if evidence:
        for item in evidence:
            label = '超时' if item.get('timed_out') else f"退出码 {item.get('exit_code', '未知')}"
            console.print(f"验证记录：{label}；完整日志已保存在会话中。", markup=False)
    else:
        console.print("本轮没有命令验证记录。")
    if backend.patches.last_export:
        console.print(f"最近导出的补丁：{backend.patches.last_export}（差异归档；请按会话模式检查实际工作区）", markup=False)
    console.print("已有修改保留；没有自动重跑工具，也没有自动撤销。")
    return True


async def display_turn(agent, history, backend, completed):
    """Only complete messages enter history; cancellation never becomes a reply."""
    original_approval = getattr(backend, "approve", None)
    approval = None
    finished = False
    metrics = getattr(backend, "ui_metrics", None) or SessionMetrics()
    metrics.begin()
    status = TurnProgress(console, metrics)
    async def ask(action):
        status.stop()
        try:
            return await original_approval(action)
        finally:
            status.start()
    if inspect.iscoroutinefunction(original_approval):
        approval = TerminalApproval(asyncio.get_running_loop(), ask,
                                    asyncio.current_task().cancel)
        backend.approve = approval
    status.start()
    try:
        async with aclosing(astream_turn(agent, history)) as events:
            async for event in events:
                if isinstance(event, ContextUpdate):
                    metrics.estimate, metrics.limit = event.estimate, event.input_limit
                    if event.compacted:
                        status.update('正在整理会话，保留你的要求…')
                elif isinstance(event, TextDelta):
                    # Render only the complete visible answer, never partial tool prose.
                    status.update('正在整理答复…')
                else:
                    history.append(event)
                    completed.append(event)
                    status.record(event)
                    if isinstance(event, AIMessage) and event.tool_calls:
                        status.update(activity(event.tool_calls[0]))
                    elif isinstance(event, ToolMessage):
                        status.update('正在检查操作结果…')
                    elif isinstance(event, AIMessage) and event.text:
                        status.stop()
                        print_answer(console, event.text, event.response_metadata.get('completion_check'))
                        if progress_entries(completed):
                            console.print('执行过程已收起，输入 /progress 查看。', style='dim')
        finished = True
    except BaseException:
        backend.cancel_all()
        raise
    finally:
        metrics.finish(finished)
        status.stop()
        if approval:
            approval.close()
            if finished:
                backend.approve = original_approval


def report_model_limit(error):
    console.print(
        f"模型调用额度已用完（{error.run_count}/{error.run_limit}）；本轮停止，会话结束。"
        "已有修改和工具记录保留；没有追加答复补全或重放工具，不代表任务完成。"
    )


def report_tool_limit(error):
    console.print(
        f"工具调用请求超过额度（累计请求 {error.run_count}，上限 {error.run_limit}，含本批未执行请求）；"
        "本批工具全部未执行，会话结束。已有修改和工具记录保留，没有重放或追加补全，不代表任务完成。"
    )


def report_time_limit(error):
    console.print(
        f"本轮已达到总时长上限（{error.seconds / 60:g} 分钟），会话结束。"
        "已有修改保留；没有重放工具或追加补全，不代表任务完成。"
    )


@app.callback()
def main(ctx: typer.Context, isolated: bool = False, task_minutes: TaskMinutes = 30,
         resume: Path | None = None, local: bool = True):
    disable_remote_tracing()
    # Propagate the root option to explicit subcommands; a subcommand option wins.
    defaults = dict(ctx.default_map or {})
    for command in ("chat", "run-task"):
        defaults[command] = {**defaults.get(command, {}), "task_minutes": task_minutes}
    ctx.default_map = defaults
    defaults['chat']['local'] = local
    if resume:
        defaults['chat']['resume'] = resume
    if ctx.invoked_subcommand is None:
        chat(isolated=isolated, task_minutes=task_minutes, resume=resume, local=local)


@app.command('inspect-session')
def inspect_session_records(session: Path, changes: Annotated[bool, typer.Option(
        help='比较当前工作区与原会话起点，不修改文件或恢复执行。')] = False):
    """查看会话状态和工具记录，不启动模型或 Docker，不恢复执行。"""
    try:
        console.print(inspect_session(session), markup=False, soft_wrap=True)
        if changes:
            from cli_rb.workspace.patches import PatchService

            snapshot = load_snapshot(session)
            console.print(f'比较工作区：{snapshot.workspace}（{snapshot.mode}）', markup=False)
            console.print('以下差异相对会话起点，仅覆盖筛选后的文件；可能包含其他程序或用户的修改。')
            patch = PatchService(snapshot).diff()
            console.print(patch or '筛选范围内没有文件内容差异。', markup=False, soft_wrap=True)
            console.print('文件差异不能证明命令完整执行或验证通过；恢复限制不变，没有重放或撤销。')
    except (ValueError, OSError) as error:
        console.print(str(error), markup=False)
        raise typer.Exit(2)


@app.command()
def explain_session(session: Path, request: Annotated[str, typer.Option(
        help='原任务要求，仅用于核对旧记录，不作为新任务执行。')]):
    """用本地模型重新说明已有证据，不恢复旧任务或执行工具。"""
    from cli_rb.interface.session_explanation import explanation_evidence, explain_evidence

    try:
        evidence = explanation_evidence(session, request)
        metadata = json.loads((session / 'conversation.json').read_text(encoding='utf-8'))
        config = ModelConfig(**metadata['model_config'])
        console.print('正在根据已有证据生成说明；独立请求最多120秒，不提供工具，不解除恢复限制。')
        reply = asyncio.run(explain_evidence(config.build_model(), evidence))
        console.print(Markdown(reply))
        console.print('以上为模型解释，不是完成证明；原会话状态未改，没有重放工具。')
    except KeyboardInterrupt:
        console.print('说明生成已取消；原会话状态未改，没有重放工具。')
        raise typer.Exit(130)
    except (ValueError, OSError, KeyError, TypeError) as error:
        console.print(f'无法生成说明：{error}；可使用 inspect-session 查看程序记录。', markup=False)
        raise typer.Exit(2)
    except Exception as error:
        console.print(f'说明请求失败：{type(error).__name__}；没有重试补全、回退模型或重放工具。')
        raise typer.Exit(1)


@app.command()
def doctor():
    """检查环境，不安装软件或启动服务。"""
    try:
        checks = inspect_environment(ModelConfig.from_env())
    except ValueError as error:
        console.print(str(error), markup=False)
        raise typer.Exit(2)
    console.print(json.dumps(checks, ensure_ascii=False, indent=2), markup=False)
    if not all(check["ok"] for check in checks):
        raise typer.Exit(1)


@app.command()
def demo():
    """无网络、无文件访问的确定性工具往返演示（不是实际模型）。"""
    console.print("模拟模型演示；不代表真实模型或容器验收。")
    agent = build_agent(DemoModel(), demo_backend())
    for message in stream_turn(agent, [HumanMessage(content=DEMO_QUESTION)]):
        render(message)


@app.command()
def smoke():
    """用真实本地模型验证结构化工具往返；不启动容器。"""
    try:
        agent = build_agent(ModelConfig.from_env().build_model(), demo_backend())
        messages = list(stream_turn(agent, [HumanMessage(content=DEMO_QUESTION)]))
        for message in messages:
            render(message)
        passed = (
            any(isinstance(m, ToolMessage) and m.name == "read_file"
                and "# CLI Rb" in str(m.content) for m in messages)
            and bool(messages)
            and isinstance(messages[-1], AIMessage)
            and bool(messages[-1].content)
            and not messages[-1].tool_calls
        )
        if not passed:
            console.print("未通过：需要成功读取演示文件，并在工具返回后给出回答。")
            raise typer.Exit(1)
        console.print("结构化工具往返通过；尚未验证流式 token、16K 上下文或显存稳定性。")
    except typer.Exit:
        raise
    except Exception as error:
        console.print(f"真实模型验证失败：{type(error).__name__}；没有启用替代模型。")
        raise typer.Exit(1)


async def approve_action(action):
    return await approval_dialog(action).run_async()


def resume_history(runner, store, agent, backend, docker):
    ensure_session_stopped(store.directory, docker)
    try:
        return runner.run(store.history(agent))
    except SessionNotReady as error:
        console.print(str(error), markup=False)
    state = runner.run(agent.graph.aget_state(store.config))
    _, missing = recovery_messages(list(state.values.get('messages', [])))
    fingerprints = store.action_fingerprints()
    patch = backend.patches.diff()
    console.print(inspect_session(store.directory), markup=False, soft_wrap=True)
    console.print('当前文件差异（相对会话起点，可能包含人工修改）：')
    console.print(patch[:12000] or '筛选范围内无差异。', markup=False, soft_wrap=True)
    if len(patch) > 12000:
        console.print('差异显示已截断；可先退出，用 inspect-session --changes 查看完整差异。')
    if missing:
        console.print('缺少已确认结果的旧请求（可能已执行，不会重放）：')
        console.print(json.dumps(missing, ensure_ascii=False, indent=2), markup=False)
    console.print('继续将保留旧记录和现有修改；旧轮次不会续跑。下一轮先读取当前文件，写操作需逐次重新确认。')
    if not typer.confirm('已检查记录和差异，同意保留现状，等待新的要求？', default=False):
        return None
    ensure_session_stopped(store.directory, docker)
    if backend.patches.diff() != patch:
        raise SessionNotReady('确认期间工作区发生变化，请重新检查后继续。')
    history = runner.run(store.recover(agent, state, fingerprints))
    backend.review_writes = True
    console.print('已建立安全继续记录；没有模型调用或工具重放。请输入新的要求。')
    return history


@app.command()
def chat(isolated: bool = False, task_minutes: TaskMinutes = 30, resume: Path | None = None,
         local: bool = True, harness: str | None = None):
    """打开项目；默认本地执行，--no-local 使用 Docker。"""
    console.rule("CLI Rb · 编码助手", style="cyan")
    try:
        if harness not in {None, 'full', 'assist'}:
            raise ValueError('--harness 仅支持 full 或 assist。')
        if isolated:
            local = False
        if resume:
            if (resume / 'execution.json').exists():
                raise ValueError('本地会话恢复尚未开放。已有修改和记录保留；请重新打开项目开始新会话。')
            local = False
            snapshot = load_snapshot(resume)
            if isolated and snapshot.mode != 'isolated':
                raise ValueError('恢复必须沿用原会话模式，不能把直接模式切换为隔离模式。')
            source = Path(snapshot.manifest['source'])
            isolated = snapshot.mode == 'isolated'
            config = None
        else:
            config = ModelConfig.from_env()
            if not config.model:
                config = replace(config, model=typer.prompt("模型 ID"))
            preferences = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'CLI Rb' / 'workspace.json'
            previous = json.loads(preferences.read_text(encoding='utf-8')) if preferences.exists() else {}
            default_repo = previous.get('repo', os.environ.get('CLI_RB_DEFAULT_REPO', str(Path.cwd())))
            source = select_workspace(console, default_repo)
        console.print(f"工作区：{source}", markup=False)
        if local:
            console.print('使用本机开发环境；常规操作自动批准，重要操作需确认。', style='dim')
        else:
            console.print('使用隔离执行环境；' + ('修改保留在副本中。' if isolated else '文件编辑直接作用于当前项目。'))
            if not typer.confirm('按此范围开始？', default=True):
                return
        if not resume:
            with console.status('正在准备工作区…'):
                snapshot = create_session(source, Path.cwd() / ".cli-rb/sessions", "isolated" if isolated else "direct")
            preferences.parent.mkdir(parents=True, exist_ok=True)
            preferences.write_text(json.dumps({'repo': str(source)}, ensure_ascii=False), encoding='utf-8')
        with SessionStore(snapshot.session, create=not resume, model_config=(
            {'base_url': config.base_url, 'model': config.model} if config else None
        ), allow_interrupted=bool(resume)) as store:
            saved_harness = store.data.get('harness', 'full')
            if resume and harness is not None and harness != saved_harness:
                raise ValueError('恢复会话必须沿用原 Harness 模式。')
            selected_harness = saved_harness if resume else harness or 'assist'
            store.data['harness'] = selected_harness
            store.save()
            config = ModelConfig(**store.data['model_config'])
            docker = None
            if local:
                backend = LocalCodingBackend(snapshot, approve=approve_action)
                backend.enable_review(config)
            else:
                image = None
                if resume:
                    image = json.loads((snapshot.session / 'container.json').read_text(encoding='utf-8'))['image']
                    if not image.startswith('sha256:') or len(image) != 71:
                        raise ValueError('原会话镜像标识无效，拒绝替换镜像。')
                with console.status('正在准备执行环境…'):
                    docker, image = prepare_image(image)
                backend = DockerCodingBackend(snapshot, docker, image, approve=approve_action,
                                              review_writes=store.data.get('review_writes', False))
            with asyncio.Runner() as runner:
                connection = AsyncSqliteSaver.from_conn_string(str(snapshot.session / 'checkpoints.sqlite'))
                saver = runner.run(connection.__aenter__())
                try:
                    agent = build_agent(config.build_model(), backend, coding=True,
                                        task_timeout_seconds=task_minutes * 60,
                                        checkpointer=saver, session=store, harness=selected_harness)
                    history = (resume_history(runner, store, agent, backend, docker) if resume
                               else runner.run(store.history(agent)))
                    if history is None:
                        return
                    console.print("准备好了。Enter 发送，Shift+Enter 换行。")
                    if backend.review_writes:
                        console.print('本会话含已核对的中断历史；修改、删除和可写命令仍需重新授权。')
                    console.print('输入 /help 查看命令。', style='dim')
                    backend.ui_metrics = SessionMetrics()
                    with chat_prompt(backend.ui_metrics) as session:
                        chat_loop(session, agent, backend, history, runner)
                finally:
                    runner.run(connection.__aexit__(None, None, None))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, sqlite3.Error) as error:
        console.print(str(error), markup=False)
        if resume and isinstance(error, SessionNotReady):
            inspect_session_records(resume)
        raise typer.Exit(2)


def chat_loop(session, agent, backend, history, runner):
    while True:
        try:
            text = session.prompt("你：")
            if not text.strip():
                continue
            if text.strip() in {'/exit', '/quit'}:
                backend.cancel_all()
                return
            if text.strip() == '/help':
                console.print(Markdown('**会话命令**\n\n- `/changes` 查看文件差异\n- `/details` 查看上次完整答复和验证说明\n- `/progress` 查看本轮执行过程\n- `/status` 查看会话位置\n- `/exit` 退出\n\nEnter 发送，Shift+Enter 换行。'))
                continue
            if text.strip() == '/details':
                reply = next((m for m in reversed(history)
                              if isinstance(m, AIMessage) and not m.tool_calls and m.text), None)
                if reply:
                    print_details(console, reply.text, reply.response_metadata.get('completion_check'))
                else:
                    console.print('还没有完成的答复。')
                continue
            if text.strip() == '/progress':
                lines = progress_entries(history)
                console.print('\n'.join(lines) if lines else '当前还没有执行过程。', markup=False)
                continue
            if text.strip() == '/changes':
                patch = backend.patches.diff()
                console.print(Syntax(patch, 'diff', word_wrap=True) if patch else '当前没有文件差异。')
                continue
            if text.strip() == '/status':
                console.print(f'工作区：{backend.snapshot.workspace}\n会话：{backend.snapshot.session}', markup=False)
                if isinstance(backend, LocalCodingBackend):
                    console.print('审批：常规操作自动通过，复杂操作独立审查，重要操作人工确认。')
                    console.print('执行：本机账户权限，无操作系统沙箱。')
                continue
            history.append(HumanMessage(content=text))
            if hasattr(backend, 'user_task'):
                backend.user_task = text
            turn_messages = []
            evidence_start = len(backend.command_evidence)

            runner.run(display_turn(agent, history, backend, turn_messages))
            report_empty_reply(turn_messages, backend, evidence_start)
        except (KeyboardInterrupt, EOFError):
            backend.cancel_all()
            console.print("对话已结束；已有修改保留，未完成输出不代表任务完成，没有重放工具。")
            return
        except ModelCallLimitExceededError as error:
            report_model_limit(error)
            return
        except ToolCallLimitExceededError as error:
            report_tool_limit(error)
            return
        except TaskTimeLimitExceeded as error:
            report_time_limit(error)
            return
        except ContextLimitExceeded as error:
            console.print(str(error), markup=False)
            return
        except SessionNotReady as error:
            console.print(str(error), markup=False)
            return
        except Exception as error:
            console.print(f"本轮失败：{type(error).__name__}。会话结束；没有自动重放工具。")
            raise typer.Exit(1)


@app.command(hidden=True)
def run_task(repo: Path, question: str, isolated: bool = False, task_minutes: TaskMinutes = 30):
    """Internal noninteractive coding entry; same agent and backend as chat."""
    backend = None
    try:
        docker, image = prepare_image()
        model = ModelConfig.from_env().build_model()
        snapshot = create_session(repo, Path.cwd() / ".cli-rb/sessions", "isolated" if isolated else "direct")
        backend = DockerCodingBackend(snapshot, docker, image)
        console.print(f"会话目录：{snapshot.session}", markup=False)
        agent = build_agent(model, backend, coding=True, task_timeout_seconds=task_minutes * 60)
        messages = []
        asyncio.run(display_turn(agent, [HumanMessage(content=question)], backend, messages))
        (snapshot.session / "task-messages.json").write_text(json.dumps([
            {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", []),
             "tool_call_id": getattr(m, "tool_call_id", None),
             "response_metadata": getattr(m, "response_metadata", {}),
             "usage_metadata": getattr(m, "usage_metadata", None)} for m in messages
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        empty_reply = report_empty_reply(messages, backend)
        (snapshot.session / "task-result.json").write_text(json.dumps({
            "end_reason": "empty_reply" if empty_reply else "model_reply",
            "command_evidence": backend.command_evidence,
            "patch": str(backend.patches.last_export) if backend.patches.last_export else None,
            "task_success": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print("本轮结束；任务是否完成需结合工具记录和补丁检查。")
    except TaskTimeLimitExceeded as error:
        report_time_limit(error)
        (snapshot.session / "task-result.json").write_text(json.dumps({
            "end_reason": "task_time_limit", "task_timeout_seconds": error.seconds,
            "command_evidence": backend.command_evidence, "task_success": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise typer.Exit(1)
    except ToolCallLimitExceededError as error:
        report_tool_limit(error)
        (snapshot.session / "task-result.json").write_text(json.dumps({
            "end_reason": "tool_call_limit", "tool_calls_requested": error.run_count,
            "tool_call_limit": error.run_limit, "pending_batch_executed": False,
            "command_evidence": backend.command_evidence, "task_success": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise typer.Exit(1)
    except ModelCallLimitExceededError as error:
        report_model_limit(error)
        (snapshot.session / "task-result.json").write_text(json.dumps({
            "end_reason": "model_call_limit", "model_calls": error.run_count,
            "model_call_limit": error.run_limit, "command_evidence": backend.command_evidence,
            "task_success": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise typer.Exit(1)
    except (Exception, KeyboardInterrupt) as error:
        if backend:
            backend.cancel_all()
        console.print(f"运行结束：{type(error).__name__}；已有修改保留，未重放工具。")
        raise typer.Exit(1)


@app.command(hidden=True)
def repo_smoke(repo: Path, question: str = "请搜索并读取代码，解释这个项目的入口和工具调用流程。"):
    """内部非交互只读验收入口；显式路径即授权该仓库的筛选副本。"""
    try:
        docker, image = prepare_image()
        model = ModelConfig.from_env().build_model()
        snapshot = create_snapshot(repo, Path.cwd() / ".cli-rb/sessions")
        console.print(f"会话目录：{snapshot.session}", markup=False)
        agent = build_agent(model, DockerReadBackend(snapshot, docker, image), workspace=True)
        messages = []
        for message in stream_turn(agent, [HumanMessage(content=question)]):
            messages.append(message)
            render(message)
        (snapshot.session / "smoke-messages.json").write_text(json.dumps([
            {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", []),
             "tool_call_id": getattr(m, "tool_call_id", None)} for m in messages
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        if not any(isinstance(m, ToolMessage) and not str(m.content).startswith("工具错误") for m in messages):
            raise RuntimeError("没有成功的工具结果，不记为仓库问答通过。")
        if not messages or not isinstance(messages[-1], AIMessage) or not messages[-1].content:
            raise RuntimeError("模型未形成最终回答。")
        console.print("只读仓库问答完成；未修改源码，也未运行仓库测试。")
    except Exception as error:
        console.print(f"仓库验证失败：{type(error).__name__}: {error}", markup=False)
        raise typer.Exit(1)
