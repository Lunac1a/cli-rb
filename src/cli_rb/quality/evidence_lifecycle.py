"""Narrow recovery of test runners that demonstrably never started."""

import re
import shlex


def test_target(command):
    if any(char in command for char in '\n\r;&|<>`$#'):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if words[:3] in (['python', '-m', 'pytest'], ['python3', '-m', 'pytest']):
        runner, args = 'pytest', words[3:]
    elif words[:2] in (['python', 'tests/runtests.py'], ['python3', 'tests/runtests.py']):
        runner, args = 'django', words[2:]
    else:
        return None
    args = [a for a in args if a not in ('-q', '-v', '-vv', '--verbose')]
    if len(args) != 1:
        return None
    target = args[0]
    if runner == 'pytest':
        if not re.fullmatch(r'tests/(?:\w+/)*\w+\.py', target):
            return None
        target = target[6:-3].replace('/', '.')
    elif not re.fullmatch(r'\w+(?:\.\w+)+', target):
        return None
    return runner, target


def startup_failure(command, result):
    target = test_target(command)
    output = result.get('output', '').strip()
    if (target and target[0] == 'pytest' and result.get('exit_code') == 1
            and not result.get('timed_out') and not result.get('junit_xml')
            and re.fullmatch(r'[^\n]+: No module named [\'"]?pytest[\'"]?', output)):
        return {'category': 'test_runner_unavailable', 'target': target[1],
                'basis': 'interpreter exited before pytest started'}
    return None


def resolved_startups(records, current):
    """Do not infer replacement from free-form scope or behavior claims."""
    active, links = [], []
    for old in records:
        failure = old.get('startup_failure')
        new = None
        if failure and old.get('purpose') != 'reproduce' and old.get('before') == old.get('after'):
            for candidate in reversed(records):
                target = test_target(candidate['command'])
                details = candidate.get('test_evidence', {})
                counts = details.get('counts') or {}
                if (candidate['kind'] == 'test' and candidate.get('purpose') != 'reproduce'
                        and candidate.get('before') == candidate.get('after') == current
                        and candidate.get('outcome') == 'passed' and candidate.get('exit_code') == 0
                        and not candidate.get('timed_out') and target
                        and target[1] == failure['target'] and counts.get('passed', 0)
                        and not counts.get('failed') and not counts.get('errors')):
                    new = candidate
                    break
        if new:
            links.append({'old_id': old['id'], 'new_id': new['id'],
                          'basis': 'runner_startup_recovered', 'target': failure['target']})
        else:
            active.append(old)
    return active, links
