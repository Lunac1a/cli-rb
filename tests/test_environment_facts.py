"""Observed capabilities must not run project code or grant command permission."""
import json
import sys
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from helpers import ScriptedModel
from test_verification import backend as controlled_backend, call
from cli_rb.workspace.container_worker import execute
from cli_rb.workspace.local_backend import LocalCodingBackend
from cli_rb.workspace.docker_backend import DockerReadBackend
from cli_rb.runtime.agent import build_agent, stream_turn

backend = controlled_backend


def test_worker_observes_command_path_without_importing_project(tmp_path, monkeypatch):
    monkeypatch.setenv('CLI_RB_COMMAND_PATH', str(Path(sys.executable).parent))
    (tmp_path / 'pytest.py').write_text("raise AssertionError('must not import project')")
    (tmp_path / '.git').mkdir()
    result = json.loads(execute(tmp_path, {'action': 'environment', 'args': {}, 'protected_paths': ['.git']}))
    assert result['git_metadata_visible'] is False
    assert result['commands']['apply_patch'] is False
    assert result['commands']['python'] is True
    assert result['pytest_in_command_python'] is True  # this test runner's installed pytest
    assert 'pytest.py' in {p.name for p in tmp_path.iterdir()}


def test_once_per_turn_static_context_and_refresh(backend):
    calls = []
    def probe():
        calls.append(1)
        return {'commands': {'rg': len(calls) > 1}, 'git_metadata_visible': False}
    backend.environment_facts = probe
    model = ScriptedModel(replies=[call('get_task_state', section='access'), AIMessage(content='Inspected.')])
    agent = build_agent(model, backend, coding=True, harness='assist')
    list(stream_turn(agent, [HumanMessage(content='Inspect')]))
    assert len(calls) == 1
    for messages in model.seen:
        assert messages[0].text.count('Turn-start environment') == 1
        assert '"rg":false' in messages[0].text
    assert not any('Turn-start environment' in m.text for m in model.seen[-1][1:])
    agent.assistance.before_agent({'messages': [HumanMessage(content='Again')]}, None)
    assert len(calls) == 2 and agent.assistance.environment['commands']['rg'] is True
    assert len((backend.snapshot.session/'environment-observations.jsonl').read_text().splitlines()) == 2


def test_probe_failure_unknown_and_local_does_not_start_python(backend, monkeypatch):
    def failed():
        raise RuntimeError('unavailable')
    backend.environment_facts = failed
    agent = build_agent(ScriptedModel(replies=[AIMessage(content='Unknown environment.')]), backend, coding=True, harness='assist')
    list(stream_turn(agent, [HumanMessage(content='Inspect')]))
    assert agent.assistance.environment == {'scope': 'unknown', 'probe_error': 'RuntimeError'}
    local = object.__new__(LocalCodingBackend)
    local.snapshot, local.env = backend.snapshot, {'PATH': ''}
    monkeypatch.setattr('subprocess.run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not run')))
    facts = local.environment_facts()
    assert facts['pytest_in_command_python'] is None
    assert facts['commands']['python'] is False


def test_container_probe_mount_is_readonly_without_write_approval(backend, monkeypatch):
    backend.review_writes = True
    backend.approve = lambda _: (_ for _ in ()).throw(AssertionError('read-only probe requested write approval'))
    observed = []
    def probe(self, action, args):
        observed.append((action, self._mount_writable))
        return '{}'
    monkeypatch.setattr(DockerReadBackend, '_call', probe)
    assert backend.environment_facts() == {}
    assert observed == [('environment', False)]
    assert backend._mount_writable is True
