"""Bounded automatic approval for common local development commands, not a sandbox."""
import re
import shlex


def command_decision(command):
    # Interpret only a single simple command. Shell expressions remain human-reviewed.
    if any(c in command for c in ('\n', '\r', ';', '|', '&', '`', '$', '>', '<', '(', ')')):
        return False, '包含复合命令或动态表达式，需要确认实际影响'
    try:
        args = [part.strip('"\'') for part in shlex.split(command, posix=False)]
    except ValueError:
        return False, '命令无法明确解析'
    if not args:
        return False, '空命令'
    name = args[0].lower()
    rest = [arg.lower() for arg in args[1:]]
    if any(re.search(r'(^|[\\/])(?:\.env[^\\/]*|\.ssh|\.aws|credentials[^\\/]*|id_rsa)([\\/]|$)', arg) for arg in rest):
        return False, '可能涉及凭据文件，需要确认'
    if name in {'get-content', 'select-string'} and any('*' in arg or '?' in arg for arg in rest):
        return False, '通配符读取范围不明确，需要确认'
    if any(re.search(r'(^|[\\/])\.\.([\\/]|$)|^[a-z]:|^[/\\]|~', arg, re.I) for arg in args[1:]):
        return False, '涉及绝对路径或项目外路径'
    if name == 'new-item' and len(rest) == 4 and rest[:3] == ['-itemtype', 'directory', '-path']:
        path = rest[3]
        if (re.fullmatch(r'[\w-]+(?:[\\/][\w-]+)*', path)
                and not any(part in {'.git', '.cli-rb'} for part in re.split(r'[\\/]', path))):
            return True, '在项目内创建普通目录'
    if name == 'git' and rest:
        if rest[0] in {'status', 'diff', 'log', 'show', 'ls-files'} and not any(
                arg.startswith(('--output', '--ext-diff', '--textconv', '--exec-path')) for arg in rest):
            return True, '查看本地 Git 状态或差异'
        return False, 'Git 写入、推送或其他仓库操作需要确认'
    if name in {'python', 'python3', 'python.exe', 'py'}:
        if len(rest) >= 2 and rest[:2] in (['-m', 'pytest'], ['-m', 'unittest'], ['-m', 'compileall']):
            return True, '执行项目测试或语法检查（项目代码具有本机权限）'
        if rest == ['--version']:
            return True, '查看解释器版本'
    if name in {'pytest', 'pytest.exe'}:
        return True, '执行项目测试（项目代码具有本机权限）'
    if name in {'node', 'node.exe'} and rest and rest[0] in {'--test', '--check', '--version'}:
        return True, '执行 JavaScript 测试或语法检查（项目代码具有本机权限）'
    if name in {'get-location', 'get-childitem', 'get-content', 'select-string', 'write-output'}:
        if not any(arg.startswith(('-stream', '-credential')) for arg in rest):
            return True, '常规本地查看操作'
    return False, '超出已识别的常规开发操作，需要确认'
