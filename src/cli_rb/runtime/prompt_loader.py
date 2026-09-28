"""Load the packaged coding instructions and insert actual runtime boundaries.

Structure adapted from the official Codex Prompting Guide; provenance and
adaptation notes live in the local collaboration documentation.
"""

from importlib.resources import files


def coding_prompt(*, mode='isolated', model_call_limit=50, tool_call_limit=100,
                  task_timeout_seconds=1800):
    if mode not in {'direct', 'isolated'}:
        raise ValueError('Unknown workspace mode')
    direct = mode == 'direct'
    workspace_rules = (
        '当前为直接模式：工具操作真实本地仓库，普通编辑立即生效。'
        'run_command 对工作区只读，需要写入的命令使用 run_write_command 申请授权；'
        '删除已有文件也需授权。不得绕过拒绝或访问被屏蔽的Git元数据、凭据；不能推送或发布。'
        if direct else
        '当前为隔离模式：读取和写入工具操作仓库副本，原仓库不挂载，修改不会自动应用到原仓库。'
        '只能在授权的副本范围内操作，不能推送或发布。'
    )
    patch_rules = (
        'show_changes / export_patch 相对会话起点展示或导出补丁；'
        + ('修改已经在本地仓库生效，导出仅作可选归档。' if direct else '导出不会自动应用补丁。')
    )
    budget_rules = (
        f'本轮上限为{model_call_limit}次模型调用、{tool_call_limit}次工具调用、'
        f'{task_timeout_seconds:g}秒，维护和补全也可能消耗预算。这些是上限，不是要用满的目标；'
        '此处不是实时剩余额度。不要假设还可以无限调用，给最终说明留出空间。'
        '程序硬限制不会因为尚未完成而自动延长。'
    )
    text = files('cli_rb').joinpath('prompts', 'coding.md').read_text(encoding='utf-8')
    for name, value in {'workspace_rules': workspace_rules, 'patch_rules': patch_rules,
                        'budget_rules': budget_rules}.items():
        marker = '{{' + name + '}}'
        if text.count(marker) != 1:
            raise ValueError(f'Coding prompt must contain exactly one {marker}')
        text = text.replace(marker, value)
    return text.strip()
