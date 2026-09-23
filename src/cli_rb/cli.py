"""Terminal UI for the first read-only integration slice."""

import json
import os

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit import PromptSession
from rich.console import Console

from cli_rb.agent import build_agent, stream_turn
from cli_rb.config import ModelConfig
from cli_rb.demo import DEMO_QUESTION, DemoModel, demo_backend
from cli_rb.doctor import inspect_environment

app = typer.Typer(invoke_without_command=True, help="CLI Rb：阶段 1，只读工具往返验证。")
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
        for call in message.tool_calls:
            console.print(f"模型请求工具：{call['name']} {call['args']}", markup=False)
        if message.content:
            console.print(f"Agent：{message.content}", markup=False)


@app.callback()
def main(ctx: typer.Context):
    disable_remote_tracing()
    if ctx.invoked_subcommand is None:
        chat()


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


@app.command()
def chat():
    """默认进入本地模型对话；当前仅开放内存演示文件。"""
    try:
        config = ModelConfig.from_env()
        if not config.model:
            config = ModelConfig(config.base_url, typer.prompt("本地模型 ID"))
        agent = build_agent(config.build_model(), demo_backend())
    except ValueError as error:
        console.print(str(error), markup=False)
        raise typer.Exit(2)
    console.print("阶段 1 / 第一块：仅可读取内存 README.md。Enter 换行，Alt+Enter 提交，Ctrl+D 退出。")
    console.print("当前对话仅保留在内存中；中断后结束会话，不提供恢复。")
    session = PromptSession(multiline=True)
    history = []
    while True:
        try:
            text = session.prompt("你：")
            if not text.strip():
                continue
            history.append(HumanMessage(content=text))
            for message in stream_turn(agent, history):
                history.append(message)
                render(message)
        except (KeyboardInterrupt, EOFError):
            console.print("对话已结束。")
            return
        except Exception as error:
            console.print(f"本轮失败：{type(error).__name__}。会话结束；没有自动重放工具。")
            raise typer.Exit(1)
