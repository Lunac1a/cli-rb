"""Interactive setup for Git-backed workspaces, before session creation."""
import os
import subprocess
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter


def git(path, *args):
    try:
        return subprocess.run(
            ['git', '-C', str(path), *args], capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=10,
            env={**os.environ, 'LC_ALL': 'C'},
        )
    except FileNotFoundError as error:
        raise ValueError('未找到 Git。请安装 Git for Windows 后重新打开软件。') from error


def select_workspace(console, default):
    prompt = PromptSession()
    while True:
        selected = prompt.prompt('项目目录（Ctrl+D 退出）：', default=default,
                                 completer=PathCompleter(only_directories=True))
        source = Path(selected.strip().strip('"')).expanduser().resolve()
        default = str(source)
        if not source.is_dir():
            console.print('这个目录不存在，请重新选择已有文件夹。', style='yellow')
            continue
        result = git(source, 'rev-parse', '--show-toplevel')
        if result.returncode:
            if 'not a git repository' not in result.stderr:
                console.print('无法读取这个 Git 项目，请处理以下原因或选择其他目录：', style='yellow')
                console.print(result.stderr.strip(), markup=False)
                continue
            console.print('这是普通文件夹。CLI Rb 需要本地 Git 项目来筛选文件和保存修改基线。')
            if not typer.confirm('初始化为本地 Git 项目并继续？只创建 .git，不提交或上传文件', default=False):
                console.print('未改动文件夹，可以重新选择目录。')
                continue
            initialized = git(source, 'init')
            if initialized.returncode:
                console.print('初始化失败，请检查目录权限或选择其他目录。', style='yellow')
                console.print(initialized.stderr.strip(), markup=False)
                continue
            console.print('本地项目已初始化。', style='green')
            return source
        root = Path(result.stdout.strip()).resolve()
        if root != source:
            console.print(f'你选择的是项目子目录，项目根目录是：{root}', markup=False)
            if not typer.confirm('改用这个项目根目录？', default=False):
                continue
        return root
