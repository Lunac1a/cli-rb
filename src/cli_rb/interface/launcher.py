"""Double-click entry; settings are selected locally, never shipped with the app."""
import json
import os
from pathlib import Path
import sys


def main():
    from rich.console import Console
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter
    console = Console()
    state = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'CLI Rb'
    state.mkdir(parents=True, exist_ok=True)
    settings = state / 'launcher.json'
    try:
        saved = json.loads(settings.read_text(encoding='utf-8')) if settings.exists() else {}
        config = os.environ.get('CLI_RB_ENV_FILE') or saved.get('config')
        if not config or not Path(config).is_file():
            console.print('欢迎使用 CLI Rb。首次启动请选择本机模型配置文件；以后会记住这个位置。')
            config = prompt('模型配置文件路径：', completer=PathCompleter()).strip().strip('"')
            if not Path(config).is_file():
                raise ValueError('配置文件不存在。可使用软件目录中的 model.env.example 填写模型服务。')
        os.environ['CLI_RB_ENV_FILE'] = str(Path(config).resolve())
        settings.write_text(json.dumps({'config': os.environ['CLI_RB_ENV_FILE']}, ensure_ascii=False), encoding='utf-8')
        # Store sessions in a writable user directory, independent of the installation.
        os.chdir(state)
        from cli_rb.interface.cli import app
        app()
    except (EOFError, KeyboardInterrupt):
        pass
    except SystemExit as error:
        if error.code:
            console.print('启动或会话未正常结束；上方保留了原因。', style='yellow')
    except Exception as error:
        console.print(f'无法启动：{error}', markup=False, style='red')
    finally:
        if sys.stdin.isatty():
            try:
                input('按 Enter 关闭窗口…')
            except (EOFError, KeyboardInterrupt):
                pass


if __name__ == '__main__':
    main()
