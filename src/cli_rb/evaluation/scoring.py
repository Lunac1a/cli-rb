"""Official SWE-bench adapter, executed only inside the scoring environment.

Requires the optional, separately installed swebench==3.0.17. This module
generates official scripts and grades captured logs; it never controls Docker.
"""

import argparse
import json
from pathlib import Path


def evidence_coverage(expected, observed, log_ok=True):
    """Missing parsed tests are an evaluation problem, not a model failure."""
    missing = sorted(set(expected) - set(observed))
    return {'valid': bool(log_ok and not missing), 'expected': len(set(expected)),
            'observed_expected': len(set(expected) & set(observed)), 'missing': missing}


def main():
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_report, get_logs_eval

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'grade'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prediction', type=Path)
    parser.add_argument('--test-log', type=Path)
    parser.add_argument('--environment-manifest', type=Path,
                        help='Attach the derived environment identity; do not label adapted scores as original.')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines()]
    row, = [row for row in rows if row['instance_id'] == args.instance_id]
    spec = make_test_spec(row)
    if args.action == 'prepare':
        content = spec.eval_script
    else:
        if not args.prediction or not args.test_log:
            parser.error('grade requires --prediction and --test-log')
        prediction = json.loads(args.prediction.read_text(encoding='utf-8'))
        if prediction['instance_id'] != args.instance_id:
            raise ValueError('Prediction instance ID mismatch')
        report = get_eval_report(spec, prediction, args.test_log, include_tests_status=True)
        observed, log_ok = get_logs_eval(spec, str(args.test_log))
        coverage = evidence_coverage([*spec.FAIL_TO_PASS, *spec.PASS_TO_PASS], observed, log_ok)
        report[args.instance_id]['evidence_coverage'] = coverage
        if not coverage['valid']:
            report[args.instance_id]['raw_official_resolved'] = report[args.instance_id]['resolved']
            report[args.instance_id]['resolved'] = None
            report[args.instance_id]['evaluation_status'] = 'evaluation_error'
        else:
            report[args.instance_id]['evaluation_status'] = 'graded'
        if args.environment_manifest:
            report[args.instance_id]['evaluation_environment'] = json.loads(
                args.environment_manifest.read_text(encoding='utf-8'))
        content = json.dumps(report, indent=2)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


if __name__ == '__main__':
    main()
