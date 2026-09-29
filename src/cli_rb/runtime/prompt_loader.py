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
        'Direct mode: tools operate on the real local repository; ordinary edits take effect immediately. '
        'run_command is read-only for the workspace; use run_write_command to request authorization for commands that write. '
        'Deleting existing files also requires authorization. Do not bypass denials or access blocked Git metadata or credentials. Do not push or publish.'
        if direct else
        'Isolated mode: read/write tools operate on a repository copy. The original repository is not mounted, and changes are not automatically applied to it. '
        'Operate only within the authorized copy. Do not push or publish.'
    )
    patch_rules = (
        'show_changes / export_patch display or export a patch relative to the session baseline. '
        + ('Changes already affect the local repository; export is optional archival.' if direct else 'Export does not automatically apply the patch.')
    )
    budget_rules = (
        f'This run allows at most {model_call_limit} model calls, {tool_call_limit} tool calls, '
        f'and {task_timeout_seconds:g} seconds; maintenance and final-answer recovery may also consume budget. These are ceilings, not targets. '
        'These numbers are not live remaining allowances. Do not assume unlimited calls; leave room for the final report. '
        'Hard limits do not automatically increase when work remains unfinished.'
    )
    text = files('cli_rb').joinpath('prompts', 'coding.md').read_text(encoding='utf-8')
    for name, value in {'workspace_rules': workspace_rules, 'patch_rules': patch_rules,
                        'budget_rules': budget_rules}.items():
        marker = '{{' + name + '}}'
        if text.count(marker) != 1:
            raise ValueError(f'Coding prompt must contain exactly one {marker}')
        text = text.replace(marker, value)
    return text.strip()
