"""Persistent, bounded evaluation batches. Solver attempts are never replayed.

Callbacks own resource limits and return JSON-serializable results. A scoring
exception is isolated; repeated scoring failures pause dispatch. In-progress
records left by a killed process require inspection, never automatic replay.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_batch(directory, task_ids, frozen_files, solve, score, *, batch_size=25,
              score_failure_limit=3, rescore=()):
    """Resume the same frozen run; explicit rescores write new attempts only.

    ``solve(id, output)`` and ``score(id, solver_result, output)`` must create
    their output directory. Solver exceptions pause the run; returned solver
    error classifications remain the callback's responsibility.
    """
    if batch_size < 1 or score_failure_limit < 1 or len(set(task_ids)) != len(task_ids):
        raise ValueError('Positive limits and unique task IDs required')
    if not frozen_files:
        raise ValueError('Freeze protocol, source, configuration and inputs before running')
    if not set(rescore) <= set(task_ids):
        raise ValueError('Rescore IDs must belong to this run')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # SQLite releases this process lock after a crash; no stale lock deletion needed.
    lock = sqlite3.connect(directory / 'dispatch-lock.sqlite', timeout=0)
    try:
        lock.execute('BEGIN EXCLUSIVE')
        path = directory / 'state.json'
        hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in frozen_files}
        definition = {'tasks': list(task_ids), 'hashes': hashes,
                      'batch_size': batch_size, 'score_failure_limit': score_failure_limit}
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['definition'] != definition:
                raise ValueError('Frozen run changed; use a new run directory')
        else:
            state = {'definition': definition, 'items': {}, 'history': []}
            save(path, state)

        def intact():
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
                   for p, h in hashes.items()):
                raise ValueError('Frozen file changed during evaluation')

        processed = failures = 0
        pause = None
        for index, iid in enumerate(task_ids):
            item = state['items'].get(iid)
            if item and item['solver']['state'] != 'returned':
                # Its solver may already have spent budget or changed files.
                pause = 'inspect_interrupted_solver:' + iid
                break
            if item and item['scores'] and iid not in rescore:
                continue
            if processed >= batch_size:
                break
            intact()
            target = directory / 'tasks' / f'{index:04d}'
            if not item:
                item = {'solver': {'state': 'running'}, 'scores': []}
                state['items'][iid] = item
                save(path, state)
                try:
                    result = solve(iid, target / 'solver')
                except Exception as exc:
                    item['solver'].update(state='error', error=f'{type(exc).__name__}: {exc}')
                    save(path, state)
                    pause = 'solver_error:' + iid
                    break
                item['solver'] = {'state': 'returned', 'result': result}
                save(path, state)
            intact()
            attempt = {'state': 'running', 'started': time.time()}
            item['scores'].append(attempt)
            save(path, state)
            try:
                result = score(iid, item['solver']['result'],
                               target / f'score-{len(item["scores"]):03d}')
                attempt.update(state='returned', result=result)
            except Exception as exc:
                result = {'state': 'evaluation_error', 'resolved': None,
                          'error': f'{type(exc).__name__}: {exc}'}
                attempt.update(state='error', result=result)
            save(path, state)
            intact()
            processed += 1
            failures = failures + 1 if result.get('state') != 'graded' else 0
            if failures >= score_failure_limit:
                pause = 'consecutive_scoring_errors'
                break
        state['history'].append({'finished': time.time(), 'processed': processed, 'pause': pause})
        # Unknown/error results count as recorded, never as successful evaluations.
        state['all_recorded'] = all(
            iid in state['items'] and state['items'][iid]['scores'] and
            state['items'][iid]['scores'][-1]['state'] in ('returned', 'error')
            for iid in task_ids)
        save(path, state)
        return state
    finally:
        lock.close()
