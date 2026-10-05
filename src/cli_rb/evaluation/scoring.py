"""Official SWE-bench adapter, executed only inside the scoring environment.

Requires the optional, separately installed swebench==3.0.17. This module
generates official scripts and grades captured logs; it never controls Docker.
"""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory


def normalize_test_log(text, repo):
    """Remove display decoration, never infer or invent test outcomes."""
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    if repo == 'sphinx-doc/sphinx':
        text = re.sub(r'^(\S+\.py::[^\r\n]+? (?:PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR))'
                      r' +\[ *\d+%\](?=\r?$)', r'\1', text, flags=re.M)
    return text


def infrastructure_errors(text):
    # Only explicit infrastructure signatures, not generic failing assertions.
    return [reason for pattern, reason in [
        ('Temporary failure in name resolution', 'network_name_resolution'),
        ('Network is unreachable', 'network_unreachable'),
    ] if pattern in text]


def evidence_coverage(expected, observed, log_ok=True):
    """Missing parsed tests are an evaluation problem, not a model failure."""
    missing = sorted(set(expected) - set(observed))
    return {'valid': bool(log_ok and not missing), 'expected': len(set(expected)),
            'observed_expected': len(set(expected) & set(observed)), 'missing': missing}


def annotate_report(report, expected, observed, log_ok, raw):
    """Keep diagnostics separate from failures of the required official tests."""
    coverage = evidence_coverage(expected, observed, log_ok)
    diagnostics = infrastructure_errors(raw)
    resolved = report['resolved']
    # Tests may intentionally capture network errors and still pass (pytest pastebin).
    # Do not let that text override complete, successful official evidence.
    problems = diagnostics if not (coverage['valid'] and resolved is True) else []
    report.update(evidence_coverage=coverage, infrastructure_errors=problems,
                  infrastructure_diagnostics=diagnostics, raw_official_resolved=resolved)
    report['evaluation_status'] = 'evaluation_error' if not coverage['valid'] or problems else 'graded'
    if report['evaluation_status'] == 'evaluation_error':
        report['resolved'] = None
    return report


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
        raw = args.test_log.read_text(encoding='utf-8')
        normalized = normalize_test_log(raw, row['repo'])
        with TemporaryDirectory() as directory:
            parser_input = Path(directory) / args.test_log.name
            parser_input.write_text(normalized, encoding='utf-8')
            report = get_eval_report(spec, prediction, parser_input, include_tests_status=True)
            observed, log_ok = get_logs_eval(spec, str(parser_input))
        annotate_report(report[args.instance_id], [*spec.FAIL_TO_PASS, *spec.PASS_TO_PASS],
                        observed, log_ok, raw)
        report[args.instance_id]['log_normalized'] = normalized != raw
        if args.environment_manifest:
            report[args.instance_id]['evaluation_environment'] = json.loads(
                args.environment_manifest.read_text(encoding='utf-8'))
        content = json.dumps(report, indent=2)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


if __name__ == '__main__':
    main()
