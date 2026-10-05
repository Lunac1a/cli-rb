"""Bounded model-facing command evidence; original receipts remain lossless."""

import json


def command_feedback(result, log_id):
    data = json.loads(result)
    data = dict(data)
    output = data.get('output', '')
    data['log_id'] = log_id
    data['full_log'] = 'read_tool_log(log_id), offset counts characters in the original JSON receipt'
    if len(output) > 12000:
        data['output'] = output[:6000] + '\n[... middle omitted; full output in log ...]\n' + output[-6000:]
        data['output_truncated'] = True
        data['output_chars'] = len(output)
    # XML is consumed from the original receipt by the verification ledger.
    if data.get('junit_xml'):
        data['junit_xml'] = '[saved in full log; use run_verification for parsed evidence]'
    return json.dumps(data, ensure_ascii=False)
