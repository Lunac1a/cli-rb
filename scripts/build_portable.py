"""Build a Windows portable folder from the locked, installed Python environment.
No user config, repository history, sessions or model credentials are copied.
"""
import json
import shutil
import sys
from pathlib import Path


def build(destination):
    root = Path(__file__).resolve().parents[1]
    base = Path(sys.base_prefix)
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / 'runtime'
    runtime.mkdir()
    for pattern in ('python.exe', 'python3*.dll', 'vcruntime*.dll', 'LICENSE.txt'):
        for p in base.glob(pattern):
            shutil.copy2(p, runtime / p.name)
    shutil.copytree(base / 'DLLs', runtime / 'DLLs')
    shutil.copytree(base / 'Lib', runtime / 'Lib',
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'idlelib', 'tkinter', 'ensurepip'))
    packages = Path(sys.prefix) / 'Lib/site-packages'
    shutil.copytree(packages, runtime / 'packages',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pth', '*.pyc', 'cli_rb', 'pytest', '_pytest', 'ruff'))
    shutil.copytree(root / 'src/cli_rb', runtime / 'packages/cli_rb',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (runtime / 'python312._pth').write_text('DLLs\nLib\npackages\nimport site\n')
    (destination / 'Start CLI Rb.cmd').write_text('@echo off\nchcp 65001 >nul\n"%~dp0runtime\\python.exe" -X utf8 -m cli_rb.interface.launcher %*\n', encoding='utf-8')
    shutil.copy2(root / '.env.example', destination / 'model.env.example')
    (destination / '使用说明.txt').write_text('双击 Start CLI Rb.cmd 启动。首次选择本机模型配置文件，然后选择项目目录；普通文件夹可按提示初始化本地 Git 项目，不提交或上传。\n无需安装 Python；需要本机安装 Git for Windows；默认使用本机开发环境，常规操作自动批准，重要或不明确的操作需确认。高级 --no-local 模式需要 Docker。\nEnter 发送，Shift+Enter 换行；/help 查看会话命令。\n此目录不包含密钥。请勿把自己的配置或会话打包分享。\n', encoding='utf-8')
    (destination / 'build.json').write_text(json.dumps({'python': sys.version, 'runtime': 'bundled CPython', 'credentials_included': False}, indent=2))
    print(destination.resolve())


if __name__ == '__main__':
    build(Path(sys.argv[1]))
