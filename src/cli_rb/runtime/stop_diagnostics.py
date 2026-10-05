"""Observed termination facts, separate from task success and inferred causes."""


def verification_gaps(report):
    gaps = []
    for item in report.get('evidence', []):
        reason = ('stale_evidence' if not item.get('current') else
                  'verification_failed' if item.get('remaining_outcome') == 'failed' else
                  'verification_unknown' if item.get('remaining_outcome') == 'unknown' else None)
        if reason:
            gaps.append({'category': reason, 'evidence_id': item['id'],
                         'log_id': item.get('log_id'), 'command': item.get('command'),
                         'observed_outcome': item.get('remaining_outcome'),
                         'test_nodes': item.get('remaining_nodes'),
                         'cause': 'unknown'})
    for behavior in report.get('behaviors', []):
        if behavior['status'] != 'evidence_current':
            gaps.append({'category': 'behavior_unverified', 'behavior_id': behavior['id'],
                         'expected': behavior['expected'], 'evidence_id': behavior.get('evidence_id')})
    return {'gaps': gaps, 'issues': report.get('issues', []),
            'completion_status': report.get('completion_status', 'unknown')}


def termination_record(error, has_final_reply, report, *, elapsed, time_limit,
                       model_calls, model_limit, tool_results, checks, final_only=False):
    name = type(error).__name__ if error else None
    trigger = {
        'TaskTimeLimitExceeded': 'time_budget',
        'ModelCallLimitExceededError': 'model_call_budget',
        'ToolCallLimitExceededError': 'tool_call_budget',
        'GraphRecursionError': 'graph_step_budget',
        'CancelledError': 'cancelled', 'GeneratorExit': 'consumer_closed',
    }.get(name, 'runtime_exception' if error else
          'model_budget_final_report' if final_only else
          'model_final_reply' if has_final_reply else 'ended_without_final_reply')
    feedback = [i for i, c in enumerate(checks) if c.get('feedback_sent')]
    first = checks[feedback[0]] if feedback else None
    last = checks[-1] if checks else None
    return {'schema_version': 1, 'trigger': trigger, 'error_type': name,
            'error_message': str(error) if error else None,
            'verification': verification_gaps(report),
            'budget': {'model_calls_used': model_calls, 'model_call_limit': model_limit,
                       'model_calls_remaining': max(0, model_limit - model_calls) if model_calls is not None else None,
                       'elapsed_seconds': elapsed, 'time_limit_seconds': time_limit,
                       'seconds_remaining': max(0, time_limit - elapsed),
                       'tool_results_observed': tool_results,
                       'tool_calls_remaining': None, 'accounting_tokens_remaining': None},
            'feedback': {'sent': len(feedback), 'completion_checks': len(checks),
                         'last_decision': last.get('feedback_decision') if last else None,
                         'new_evidence_ids_after_first_feedback': sorted(set(last.get('evidence_ids', [])) - set(first.get('evidence_ids', []))) if first else [],
                         'issues_before': len(first['issues']) if first else None,
                         'issues_after': len(last['issues']) if last else None},
            'task_success': None,
            'interpretation': 'Observed stop trigger only; verification failure does not classify environment vs product failure. New evidence is not proof of useful progress. Null budget fields are unknown.'}
