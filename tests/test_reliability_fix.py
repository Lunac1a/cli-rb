from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from helpers import ScriptedModel
from test_verification import backend, call  # noqa: F401
from cli_rb.runtime.agent import build_agent, stream_turn
from cli_rb.evaluation.scoring import normalize_test_log, infrastructure_errors
from cli_rb.evaluation.execution import run_scoring_script


def test_coding_last_request_reports_within_budget(backend):  # noqa: F811
    model = ScriptedModel(replies=[
        call('read_file', path='app.py'), call('read_file', path='app.py'),
        AIMessage(content='Inspected only; no changes or tests.'),
    ])
    agent = build_agent(model, backend, coding=True, model_call_limit=3)
    events = list(stream_turn(agent, [HumanMessage(content='Inspect')]))
    assert 'Inspected only' in events[-1].text
    assert agent.requests.calls == 3
    assert agent.requests.records[-1]['final_only']
    assert 'final request within the existing budget' in model.seen[-1][-1].text
    assert sum(len(m.tool_calls) for m in events if isinstance(m, AIMessage)) == 2


def test_capacity_applies_to_preimported_ledger_and_patch(backend):  # noqa: F811
    # The ledger imported inventory before configuration, exactly as in the failed run.
    path = backend.snapshot.workspace / 'fixture.bin'
    with path.open('wb') as stream:
        stream.truncate(65 * 1024 * 1024)
    backend.snapshot.manifest['max_total_bytes'] = 128 * 1024 * 1024
    backend.snapshot.manifest['max_file_bytes'] = 128 * 1024 * 1024
    backend.patches.max_file_bytes = 128 * 1024 * 1024
    backend.verification.begin_turn()
    assert backend.verification.fingerprint()
    # Unchanged native fixture is baseline data, not an exported binary edit.
    backend.patches._before['fixture.bin'] = path.read_bytes()
    assert backend.patches.diff() == ''


def test_colored_sphinx_and_network_diagnostic():
    raw = 'tests/test_a.py::test_one \x1b[32mPASSED\x1b[0m\x1b[33m [ 16%]\x1b[0m\n'
    assert normalize_test_log(raw, 'sphinx-doc/sphinx') == 'tests/test_a.py::test_one PASSED\n'
    assert infrastructure_errors('Temporary failure in name resolution') == ['network_name_resolution']
    assert infrastructure_errors('AssertionError: wrong value') == []


def test_setup_failure_prevents_tests(monkeypatch, tmp_path):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1 if 'exec' in args else 0)
    monkeypatch.setattr('cli_rb.evaluation.execution.subprocess.run', run)
    script = '#!/bin/bash\ncd /testbed\ngit config --global --add safe.directory /testbed\npip install .\ngit apply -v - <<EOF\nEOF\npytest\n'
    result = run_scoring_script('docker', 'container', script, tmp_path/'score')
    assert result['phase'] == 'setup'
    assert len(calls) == 2
    assert calls[-1][1]['timeout'] == 1800


def test_budget_checkpoint_preserves_tool_work_and_reserves_final(backend):  # noqa: F811
    model = ScriptedModel(replies=[
        *[call('read_file', path='app.py') for _ in range(6)],
        AIMessage(content='Inspected only; no changes or tests.'),
    ])
    agent = build_agent(model, backend, coding=True, model_call_limit=7)
    events = list(stream_turn(agent, [HumanMessage(content='Inspect')]))
    assert not any('Budget checkpoint:' in m.text for m in model.seen[1])
    assert '5 model requests remain' in model.seen[2][-1].text
    assert '2 model requests remain' in model.seen[5][-1].text
    assert 'final request within the existing budget' in model.seen[6][-1].text
    assert sum(len(m.tool_calls) for m in events if isinstance(m, AIMessage)) == 6
    assert agent.requests.calls == 7
