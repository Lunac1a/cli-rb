"""Terminal input and rendering, separate from agent and workspace execution."""

import json
import os
import subprocess
import signal
from contextlib import contextmanager
from pathlib import Path

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit import PromptSession
from rich.console import Console

from cli_rb.agent import build_agent, stream_turn, has_reply
from cli_rb.config import ModelConfig
from cli_rb.demo import DEMO_QUESTION, DemoModel, demo_backend
from cli_rb.doctor import inspect_environment
from cli_rb.docker_backend import DockerReadBackend, prepare_image
from cli_rb.snapshot import create_snapshot, create_session
from cli_rb.coding_backend import DockerCodingBackend

app = typer.Typer(invoke_without_command=True, help="CLI Rb：本地工作区编码助手。")
console = Console()


def disable_remote_tracing():
    # Local by default, even when the parent shell has tracing enabled.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"


def render(message):
    if isinstance(message, ToolMessage):
        console.print(f"工具返回 [{message.name}]：{message.content}", markup=False)
    elif isinstance(message, AIMessage):
        recovery = message.response_metadata.get("final_reply_recovery")
        if recovery:
            console.print(f"程序：已尝试一次无工具的答复补全（{recovery}）；没有重跑工具。")
        for call in message.tool_calls:
            console.print(f"模型请求工具：{call['name']} {call['args']}", markup=False)
        if message.content:
            console.print(f"Agent：{message.content}", markup=False)


def report_empty_reply(messages, backend, evidence_start=0):
    if messages and has_reply(messages[-1]):
        return False
    console.print("模型未返回最终文字答复；以下为程序记录，不代表模型已确认需求全部完成。")
    evidence = backend.command_evidence[evidence_start:]
    if evidence:
        for item in evidence:
            console.print(json.dumps(item, ensure_ascii=False), markup=False)
    else:
        console.print("本轮没有命令验证记录。")
    if backend.patches.last_export:
        console.print(f"最近导出的补丁：{backend.patches.last_export}（差异归档；请按会话模式检查实际工作区）", markup=False)
    console.print("已有修改保留；没有自动重跑工具，也没有自动撤销。")
    return True


@contextmanager
def cancel_on_interrupt(backend):
    previous = signal.getsignal(signal.SIGINT)

    def cancel(signum, frame):
        try:
            backend.cancel_all()
        finally:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@app.callback()
def main(ctx: typer.Context, isolated: bool = False):
    disable_remote_tracing()
    if ctx.invoked_subcommand is None:
        chat(isolated=isolated)


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


def approve_action(action):
    console.print(action, markup=False)
    return typer.confirm("允许此操作？", default=False)


@app.command()
def chat(isolated: bool = False):
    """选择仓库并确认范围后，在 Docker 中调查、修改和验证本地工作区。"""
    try:
        config = ModelConfig.from_env()
        if not config.model:
            config = ModelConfig(config.base_url, typer.prompt("本地模型 ID"))
        source = Path(typer.prompt("Git 仓库目录", default=str(Path.cwd()))).expanduser()
        if isolated:
            console.print("隔离模式：只修改筛选后的副本，原仓库不挂载；补丁可选导出。")
        else:
            console.print("直接模式：普通编辑立即写入所选本地仓库；保存会话起点作为差异基线。")
            console.print("删除已有文件、允许命令写入仓库需要单独授权；普通命令只读访问仓库。")
        console.print("命令在禁网 Linux 容器内运行；镜像提供 Python 标准库。Git 元数据和常见凭据不可访问。")
        if not typer.confirm("按此范围开始？", default=True):
            return
        docker, image = prepare_image()
        snapshot = create_session(source, Path.cwd() / ".cli-rb/sessions", "isolated" if isolated else "direct")
        console.print(f"已保存基线 {len(snapshot.manifest['files'])} 个文件；排除 {len(snapshot.manifest['excluded'])} 项。")
        console.print(f"会话和筛选清单：{snapshot.session}", markup=False)
        backend = DockerCodingBackend(snapshot, docker, image, approve=approve_action)
        agent = build_agent(config.build_model(), backend, coding=True)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        console.print(str(error), markup=False)
        raise typer.Exit(2)
    console.print("编码会话。Enter 换行，Alt+Enter 提交（也可先 Esc 再 Enter），Ctrl+D 退出。")
    console.print("当前对话仅保留在内存中；中断后结束会话，不提供恢复。")
    session = PromptSession(multiline=True)
    history = []
    while True:
        try:
            text = session.prompt("你：")
            if not text.strip():
                continue
            history.append(HumanMessage(content=text))
            turn_messages = []
            evidence_start = len(backend.command_evidence)
            console.print("Agent 正在处理…")
            with cancel_on_interrupt(backend):
                for message in stream_turn(agent, history):
                    history.append(message)
                    turn_messages.append(message)
                    render(message)
            report_empty_reply(turn_messages, backend, evidence_start)
        except (KeyboardInterrupt, EOFError):
            backend.cancel_all()
            console.print("对话已结束。")
            return
        except Exception as error:
            console.print(f"本轮失败：{type(error).__name__}。会话结束；没有自动重放工具。")
            raise typer.Exit(1)


@app.command(hidden=True)
def run_task(repo: Path, question: str, isolated: bool = False):
    """Internal noninteractive coding entry; same agent and backend as chat."""
    backend = None
    try:
        docker, image = prepare_image()
        model = ModelConfig.from_env().build_model()
        snapshot = create_session(repo, Path.cwd() / ".cli-rb/sessions", "isolated" if isolated else "direct")
        backend = DockerCodingBackend(snapshot, docker, image)
        console.print(f"会话目录：{snapshot.session}", markup=False)
        agent = build_agent(model, backend, coding=True)
        messages = []
        with cancel_on_interrupt(backend):
            for message in stream_turn(agent, [HumanMessage(content=question)]):
                messages.append(message)
                render(message)
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
