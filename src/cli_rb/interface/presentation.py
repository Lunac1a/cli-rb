"""Quiet terminal presentation; full evidence stays in session records."""
from pathlib import PurePath
import json
import time
from rich.console import Group
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


def activity(call):
    name = call.get('name', '')
    labels = {
        'read_file': '正在阅读文件', 'list_directory': '正在查看目录',
        'find_files': '正在查找文件', 'search_files': '正在搜索代码',
        'edit_file': '正在修改文件', 'write_file': '正在写入文件',
        'create_file': '正在创建文件', 'delete_file': '正在删除文件',
        'run_write_command': '正在执行已授权的命令',
        'run_command': '正在执行命令', 'run_verification': '正在运行验证',
        'review_work': '正在检查改动', 'show_changes': '正在核对差异',
        'write_todos': '正在安排工作步骤', 'read_tool_log': '正在检查运行日志',
        'register_behavior': '正在明确验收要求', 'bind_verification': '正在核对验证结果',
        'replace_verification': '正在更新验证记录', 'read_history': '正在回顾会话',
    }
    label = labels.get(name, '正在处理任务')
    path = call.get('args', {}).get('path')
    if isinstance(path, str) and name in {'read_file', 'edit_file', 'write_file'}:
        label += ' · ' + PurePath(path).name.replace('\\', '/').split('/')[-1]
    return label + '…'


def progress_entries(messages):
    """Describe observable tool events, never intermediate model prose or arguments."""
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    entries = []
    for message in messages:
        if isinstance(message, HumanMessage):
            entries = []
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                line = activity(call)
                if not entries or entries[-1] != line:
                    entries.append(line)
        elif isinstance(message, ToolMessage):
            try:
                result = json.loads(message.text)
            except (ValueError, TypeError):
                result = {}
            if not isinstance(result, dict):
                result = {}
            if message.status == 'error' or message.text.startswith('工具错误'):
                entries.append('操作未成功，正在检查原因。')
            elif result.get('timed_out'):
                entries.append('命令超时，正在检查下一步。')
            elif result.get('exit_code') not in (None, 0):
                entries.append('命令未成功，正在检查原因和可用的替代方式。')
            elif result.get('exit_code') == 0:
                entries.append('命令已结束，正在核对输出；退出成功不代表测试全部通过。')
    return entries


class SessionMetrics:
    def __init__(self):
        self.estimate = None
        self.limit = 0
        self.started = None
        self.duration = 0
        self.phase = '就绪'

    def begin(self):
        self.started = time.monotonic()
        self.duration = 0
        self.phase = '处理中'

    def finish(self, successful):
        if self.started is not None:
            self.duration = time.monotonic() - self.started
            self.started = None
        self.phase = '已结束' if successful else '已停止'

    def text(self):
        seconds = int(time.monotonic() - self.started if self.started is not None else self.duration)
        context = '上下文：等待统计'
        if self.estimate is not None:
            context = f'最近请求上下文估算：{self.estimate:,} tokens'
            if self.limit:
                context += f' / {self.limit:,} 输入预算（{self.estimate / self.limit:.0%}）'
        return f'{self.phase}  ·  {context}  ·  本轮 {seconds // 60:02d}:{seconds % 60:02d}'


class TurnProgress:
    """Transient stage log, collapsed before the final answer or an approval prompt."""
    def __init__(self, console, metrics=None):
        self.messages = []
        self.metrics = metrics or SessionMetrics()
        self.label = '正在理解你的要求…'
        self.live = Live(console=console, transient=True, refresh_per_second=4,
                         get_renderable=self.render)

    def render(self):
        lines = progress_entries(self.messages)
        return Group(*(Text(line, style='dim') for line in lines[-6:]),
                     Spinner('dots', text=Text(self.label)),
                     Text(self.metrics.text(), style='cyan', justify='right'))

    def start(self):
        self.live.start()
        self.update(self.label)

    def stop(self):
        self.live.stop()

    def update(self, label):
        self.label = label
        self.live.update(self.render())

    def record(self, message):
        self.messages.append(message)
        self.update(self.label)


def print_answer(console, text, report=None):
    # Internal audit text remains in the session; the terminal uses plain-language status.
    answer, separator, details = text.partition('[程序核对]')
    if answer.startswith('[完成状态]'):
        answer = answer.partition('\n')[2].lstrip()
    console.print()
    console.print(Markdown(answer.strip(), code_theme='monokai'))
    if separator:
        incomplete = ((report or {}).get('status') == 'needs_attention'
                      or '部分完成' in details.split('。')[0])
        summary = ('部分完成：还有未通过或未能确认的检查，暂不能确认功能全部正常。' if incomplete else
                   '已记录检查结果；这些检查不保证覆盖所有使用情况。')
        console.print(Panel(Text(summary + '输入 /details 查看测试情况。'),
                            title='验证状态', border_style='dim'))


def print_details(console, text, report=None):
    print_answer(console, text, report)
    evidence = (report or {}).get('evidence', [])
    if not evidence:
        console.print('没有可展示的检查明细。原始记录保存在 /status 显示的会话目录。')
        return
    labels = {'passed': '通过', 'failed': '未通过', 'unknown': '结果未能确认'}
    console.print('\n测试情况', style='bold')
    for index, item in enumerate(evidence, 1):
        outcome = item.get('remaining_outcome', item.get('outcome'))
        label = labels.get(outcome, '结果未能确认')
        if item.get('timed_out'):
            label = '运行超时'
        if item.get('current') is False:
            label += '（修改前的结果，不能代表当前代码）'
        console.print(f'{index}. {label}', markup=False)
    console.print('具体检查范围与限制见上方答复；原始日志保存在会话目录。', style='dim')
