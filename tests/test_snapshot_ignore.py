import subprocess

import pytest

from cli_rb.workspace.snapshot import create_snapshot
from test_verification import ControlledBackend


def test_evaluation_snapshot_uses_git_blob_line_endings(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo/'.gitattributes').write_text('*.py text=auto\n')
    (repo/'app.py').write_bytes(b'value = 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    (repo/'app.py').write_bytes(b'value = 1\r\n')
    snapshot = create_snapshot(repo, tmp_path/'sessions', normalize_git_eol=True)
    assert (snapshot.workspace/'app.py').read_bytes() == b'value = 1\n'
    assert (repo/'app.py').read_bytes() == b'value = 1\r\n'


def test_generated_ignored_files_do_not_break_patch_or_stale_evidence(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo/'.gitignore').write_text('result_images/\n')
    (repo/'app.py').write_text('value = 1\n')
    snapshot = create_snapshot(repo, tmp_path/'sessions')
    backend = ControlledBackend(snapshot, 'docker', 'fake')
    fingerprint = backend.verification.fingerprint()
    generated = snapshot.workspace/'result_images'
    generated.mkdir()
    (generated/'plot.png').write_bytes(b'\0PNG')
    assert backend.verification.fingerprint() == fingerprint
    assert backend.patches.diff() == ''
    # Changing task rules cannot hide a new source file.
    (snapshot.workspace/'.gitignore').write_text('result_images/\nnew.py\n')
    (snapshot.workspace/'new.py').write_text('value = 2\n')
    assert 'new.py' in backend.patches.diff()


def test_tracked_file_under_ignored_directory_remains_protected(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    (repo/'.gitignore').write_text('result_images/\n')
    (repo/'result_images').mkdir()
    (repo/'result_images/tracked.png').write_bytes(b'\0old')
    subprocess.run(['git', '-C', str(repo), 'add', '-f', '.gitignore', 'result_images/tracked.png'],
                   check=True, capture_output=True)
    snapshot = create_snapshot(repo, tmp_path/'sessions')
    backend = ControlledBackend(snapshot, 'docker', 'fake')
    (snapshot.workspace/'result_images/generated.png').write_bytes(b'\0generated')
    assert backend.patches.diff() == ''
    (snapshot.workspace/'result_images/tracked.png').write_bytes(b'\0changed')
    with pytest.raises(ValueError, match='二进制'):
        backend.patches.diff()
