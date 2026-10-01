"""Apply the original repository's ignore rules to new isolated-workspace files."""
from pathlib import PurePosixPath
import os
import subprocess

from cli_rb.workspace.snapshot import excluded


def snapshot_ignored_paths(snapshot):
    if snapshot.mode == 'direct':
        from cli_rb.workspace.live_workspace import ignored_paths
        return ignored_paths(snapshot.workspace) - snapshot.manifest['files'].keys()
    # Use immutable baseline rules, not .gitignore edits made during the task.
    baseline = snapshot.session / 'baseline'
    git_dir = snapshot.session / 'ignore-rules.git'
    if not git_dir.exists():
        subprocess.run(['git', 'init', '--bare', str(git_dir)],
                       capture_output=True, check=True, timeout=10)
    candidates = []
    for directory, dirs, files in os.walk(snapshot.workspace, followlinks=False):
        relative = os.path.relpath(directory, snapshot.workspace)
        prefix = '' if relative == '.' else relative.replace(os.sep, '/') + '/'
        dirs[:] = [d for d in dirs if not excluded(PurePosixPath(prefix + d))]
        candidates.extend(prefix + name for name in [*dirs, *files]
                          if not excluded(PurePosixPath(prefix + name))
                          and prefix + name not in snapshot.manifest['files'])
    if not candidates:
        return set()
    result = subprocess.run(
        ['git', '-c', 'core.bare=false', '-c', 'core.excludesFile=',
         '--git-dir=' + str(git_dir), '--work-tree=' + str(baseline),
         'check-ignore', '--no-index', '-z', '--stdin'],
        input=('\0'.join(candidates) + '\0').encode(), capture_output=True, timeout=10,
        cwd=baseline)
    if result.returncode not in (0, 1):
        raise RuntimeError('Could not evaluate baseline ignore rules')
    protected = set(snapshot.manifest['files'])
    protected.update(str(parent) for name in snapshot.manifest['files']
                     for parent in PurePosixPath(name).parents)
    return set(filter(None, result.stdout.decode().split('\0'))) - protected
