import json
import sqlite3

import pytest

from cli_rb.evaluation.batch import run_batch
from cli_rb.evaluation.scoring import annotate_report


def test_passed_network_capture_does_not_mask_real_environment_failure():
    raw = 'captured stderr: Temporary failure in name resolution'
    passed = annotate_report({'resolved': True}, ['a'], {'a': 'PASSED'}, True, raw)
    assert passed['resolved'] is True and passed['evaluation_status'] == 'graded'
    assert passed['infrastructure_diagnostics'] == ['network_name_resolution']
    failed = annotate_report({'resolved': False}, ['a'], {'a': 'FAILED'}, True, raw)
    assert failed['resolved'] is None and failed['raw_official_resolved'] is False
    missing = annotate_report({'resolved': True}, ['a', 'b'], {'a': 'PASSED'}, True, raw)
    assert missing['resolved'] is None


def test_batches_resume_without_replaying_solver_and_rescore_preserves_error(tmp_path):
    frozen = tmp_path / 'protocol.json'
    frozen.write_text('{}')
    calls = []
    def solve(iid, output):
        output.mkdir(parents=True)
        calls.append(iid)
        return {'patch': str(output / 'model.patch')}
    def score(iid, result, output):
        output.mkdir(parents=True)
        if iid == 'a' and output.name == 'score-001':
            raise RuntimeError('missing build dependency')
        return {'state': 'graded', 'resolved': iid != 'b'}
    args = (tmp_path/'run', ['a', 'b', 'c'], [frozen], solve, score)
    first = run_batch(*args, batch_size=2)
    assert calls == ['a', 'b'] and not first['all_recorded']
    assert first['items']['a']['scores'][0]['state'] == 'error'
    second = run_batch(*args, batch_size=2)
    assert calls == ['a', 'b', 'c'] and second['all_recorded']
    third = run_batch(*args, batch_size=2, rescore=['a'])
    assert calls == ['a', 'b', 'c']
    assert len(third['items']['a']['scores']) == 2
    frozen.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Frozen run changed'):
        run_batch(*args, batch_size=2)


def test_crash_during_solver_is_never_replayed(tmp_path):
    frozen = tmp_path/'frozen'
    frozen.write_text('v1')
    calls = []
    def solve(iid, output):
        calls.append(iid)
        raise KeyboardInterrupt()
    args = (tmp_path/'run', ['a'], [frozen], solve, lambda *args: None)
    with pytest.raises(KeyboardInterrupt):
        run_batch(*args)
    state = run_batch(*args)
    assert calls == ['a']
    assert state['history'][-1]['pause'] == 'inspect_interrupted_solver:a'
    assert json.loads((tmp_path/'run/state.json').read_text())['items']['a']['solver']['state'] == 'running'


def test_repeated_scoring_failure_pauses_remaining_tasks(tmp_path):
    frozen = tmp_path/'frozen'
    frozen.write_text('v1')
    calls = []
    def solve(iid, output):
        calls.append(iid)
        return {}
    state = run_batch(tmp_path/'run', ['a', 'b', 'c', 'd'], [frozen], solve,
                      lambda *args: {'state': 'evaluation_error', 'resolved': None})
    assert calls == ['a', 'b', 'c']
    assert state['history'][-1]['pause'] == 'consecutive_scoring_errors'


def test_scoring_crash_can_be_explicitly_rescored_without_solving_again(tmp_path):
    frozen = tmp_path/'frozen'
    frozen.write_text('v1')
    calls = []
    def solve(iid, output):
        calls.append(iid)
        return {'patch': 'saved'}
    def crash(*args):
        raise KeyboardInterrupt()
    args = (tmp_path/'run', ['a'], [frozen], solve)
    with pytest.raises(KeyboardInterrupt):
        run_batch(*args, crash)
    resumed = run_batch(*args, crash)
    assert calls == ['a'] and not resumed['all_recorded']
    recovered = run_batch(*args, lambda *args: {'state': 'graded', 'resolved': True},
                          rescore=['a'])
    assert recovered['all_recorded'] and calls == ['a']
    assert recovered['items']['a']['scores'][0]['state'] == 'running'


def test_second_dispatcher_cannot_start_while_first_holds_lock(tmp_path):
    frozen = tmp_path/'frozen'
    frozen.write_text('v1')
    directory = tmp_path/'run'
    directory.mkdir()
    connection = sqlite3.connect(directory/'dispatch-lock.sqlite')
    connection.execute('BEGIN EXCLUSIVE')
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            run_batch(directory, ['a'], [frozen], lambda *a: pytest.fail('solver started'),
                      lambda *a: None)
    finally:
        connection.close()
