"""Extract reported results, without treating output as proof of coverage."""

import re
from collections import Counter
from xml.etree import ElementTree


STATUSES = ('passed', 'failed', 'skipped', 'xfailed', 'xpassed', 'errors')
LABELS = dict(zip(STATUSES, ('通过', '失败', '跳过', '预期失败', '意外通过', '错误')))


def parse_junit(result):
    """Read framework-generated testcase identities, never infer them from selectors."""
    root = ElementTree.fromstring(result['junit_xml'])
    if root.tag not in {'testsuite', 'testsuites'}:
        raise ValueError('Unsupported JUnit root')
    suites = [s for s in root.iter('testsuite') if not s.findall('testsuite')]
    cases = [case for suite in suites for case in suite.findall('testcase')]
    nodes, counts = {}, dict.fromkeys(STATUSES, 0)
    for case in cases:
        key = 'junit:' + case.attrib.get('classname', '') + '::' + case.attrib['name']
        if case.find('error') is not None:
            status = 'errors'
        elif case.find('failure') is not None:
            status = 'failed'
        elif case.find('skipped') is not None:
            status = 'xfailed' if case.find('skipped').get('type') == 'pytest.xfail' else 'skipped'
        else:
            status = 'passed'
        counts[status] += 1
        nodes[key] = status
    complete = bool(cases and len(nodes) == len(cases) and
                    sum(int(s.attrib['tests']) for s in suites) == len(cases) and
                    not result.get('timed_out') and result.get('exit_code') in (0, 1) and
                    (result.get('exit_code') == 0 or counts['failed'] or counts['errors']))
    return {'framework': 'pytest', 'source': 'junit', 'counts': counts,
            'total': len(cases), 'nodes': nodes, 'nodes_complete': complete}


def parse_test_evidence(result):
    if result.get('junit_xml'):
        try:
            return parse_junit(result)
        except (ElementTree.ParseError, ValueError, KeyError):
            # Preserve the raw report for inspection; text remains a fallback.
            pass
    output = re.sub(r'\x1b\[[0-9;]*m', '', result.get('output', ''))
    counts, total, framework = None, None, None
    summaries = []
    for line in output.splitlines():
        line = line.strip().strip('=').strip()
        if re.fullmatch(r'\d+ (?:passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?)'
                        r'(?:, \d+ (?:passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?))*'
                        r'(?: in .+)?', line):
            summaries.append(line)
    if len(summaries) == 1:
        framework = 'pytest'
        counts = dict.fromkeys(STATUSES, 0)
        for number, status in re.findall(r'(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)\b', summaries[0]):
            counts['errors' if status == 'error' else status] = int(number)
        # A teardown error may belong to an already counted passing test.
        total = None if counts['errors'] else sum(counts.values())
    runs = re.findall(r'^Ran (\d+) tests? in [^\n]*$', output, re.M)
    endings = re.findall(r'^(OK|FAILED)(?: \(([^\n]*)\))?\s*$', output, re.M)
    if not summaries and len(runs) == len(endings) == 1:
        framework, total = 'unittest', int(runs[0])
        counts = dict.fromkeys(STATUSES, 0)
        aliases = {'failures': 'failed', 'errors': 'errors', 'skipped': 'skipped',
                   'expected failures': 'xfailed', 'unexpected successes': 'xpassed'}
        fields = endings[0][1]
        pairs = re.findall(r'([a-z ]+)=(\d+)', fields)
        if all(key.strip() in aliases for key, _ in pairs):
            for key, number in pairs:
                counts[aliases[key.strip()]] = int(number)
            counts['passed'] = total - sum(counts.values())
            # Subtest errors may outnumber test methods: don't invent a pass count.
            if counts['passed'] < 0 or (endings[0][0] == 'FAILED' and not any(
                    counts[key] for key in ('failed', 'errors', 'xpassed'))):
                counts = None
            elif endings[0][0] == 'FAILED':
                # Several failing subtests can belong to one method.
                counts['passed'] = None
        else:
            counts = None

    nodes = {}
    duplicate = False
    if framework == 'pytest':
        for match in re.finditer(
                r'^(\S+\.py::[^\n]+?)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)'
                r'(?:\s+\[[^\n]*\])?(?:\s+\([^\n]*\))?\s*$', output, re.M):
            node, status = match.groups()
            duplicate |= node in nodes
            nodes[node] = {'XFAIL': 'xfailed', 'XPASS': 'xpassed', 'ERROR': 'errors'}.get(status, status.lower())
    complete = bool(nodes and counts and not duplicate and not result.get('timed_out') and
                    result.get('exit_code') in (0, 1) and
                    (result.get('exit_code') == 0 or counts['failed'] or counts['errors']) and
                    all(Counter(nodes.values())[s] == counts[s] for s in STATUSES))
    return {'framework': framework, 'counts': counts, 'total': total,
            'nodes': nodes, 'nodes_complete': complete}


def resolve_evidence(records, reconciliations=()):
    """Return active portions and explicit replacement links; keep the raw ledger intact.

    Only complete node receipts can replace node receipts. Summary-only/legacy runs
    remain unresolved rather than guessing coverage from scope or command strings.
    """
    active, replacements = [], []
    for record in records:
        evidence = record.get('test_evidence', {})
        nodes = dict(evidence.get('nodes', {})) if evidence.get('nodes_complete') else None
        stable = record.get('before') == record.get('after') and record.get('after') is not None
        if stable and not record.get('timed_out'):
            for previous in active:
                old = previous['record']
                tested = {node for node, status in (nodes or {}).items()
                          if status in {'passed', 'failed', 'errors'} or
                          status == (previous['nodes'] or {}).get(node) == 'xfailed'}
                overlap = set(previous['nodes'] or {}) & tested
                same_check = (record['kind'] == old['kind'] == 'check'
                              and record['command'] == old['command'])
                # Reproduction is historical only when a current successful receipt
                # checks the same declared behavior and assertion contract.
                behavior_link = (old.get('purpose') == 'reproduce' and
                                 old.get('expected_failure_observed') and
                                 record.get('purpose', 'verify') != 'reproduce' and
                                 record['outcome'] in {'passed', 'command_succeeded'} and
                                 bool(record.get('assertion')) and
                                 bool(old.get('behavior_contracts')) and
                                 all(record.get('behavior_contracts', {}).get(key) == value
                                     for key, value in old['behavior_contracts'].items()))
                if behavior_link:
                    previous['replaced'] = True
                    replacements.append({'old_id': old['id'], 'new_id': record['id'],
                                         'behaviors': list(old['behavior_contracts']),
                                         'basis': 'declared_same_behavior_with_observed_results'})
                    continue
                if record['kind'] == old['kind'] == 'test' and overlap:
                    for node in overlap:
                        del previous['nodes'][node]
                    replacements.append({'old_id': old['id'], 'new_id': record['id'],
                                         'nodes': sorted(overlap)})
                elif same_check:
                    previous['replaced'] = True
                    replacements.append({'old_id': old['id'], 'new_id': record['id'],
                                         'command': record['command']})
        active.append({'record': record, 'nodes': nodes, 'replaced': False})
        active = [item for item in active if not item['replaced'] and item['nodes'] != {}]
    for link in reconciliations:
        for previous in active:
            if previous['record']['id'] == link['old_id'] and link.get('check_migration'):
                previous['replaced'] = True
                replacements.append({**link, 'basis': 'explicit_check_to_test_claim'})
                continue
            if previous['record']['id'] == link['old_id'] and previous['nodes'] is not None:
                overlap = set(previous['nodes']) & set(link['old_nodes'])
                if overlap:
                    for node in overlap:
                        del previous['nodes'][node]
                    replacements.append({**link, 'nodes': sorted(overlap),
                                         'basis': 'explicit_test_replacement_claim'})
    return [item for item in active if item['nodes'] != {} and not item['replaced']], replacements


def evidence_summary(report):
    lines = []
    for item in report.get('evidence', []):
        counts = item.get('test_evidence', {}).get('counts')
        amounts = '、'.join(f'{LABELS[s]} {counts[s]}' for s in STATUSES if counts[s]) if counts else '测试数量未知'
        if counts and not any(counts.values()):
            amounts = '无已识别测试结果'
        if counts and counts['passed'] is None:
            amounts += '、通过数量未知'
        if item['kind'] == 'check':
            amounts = '命令检查，不计作测试通过'
        version = '当前版本' if item['current'] else '版本过期或执行期间发生变化'
        nodes = item['remaining_nodes']
        coverage = '；实际节点未完整识别'
        if nodes is not None:
            shown = ', '.join(f'{node}={status}' for node, status in list(nodes.items())[:8])
            coverage = f'；仍关联 {len(nodes)} 个节点：{shown}'
            if len(nodes) > 8:
                coverage += '（完整节点见证据记录）'
        lines.append(f"- {item['id']}：原运行 {amounts}；仍关联部分 {item['remaining_outcome']}；{version}{coverage}")
    if report.get('replacements'):
        lines.append(f"新旧证据有 {len(report['replacements'])} 条替代关系，原日志保留；上述计数按每次运行列出，不跨运行相加。")
        for link in report['replacements']:
            if link.get('basis') in {'explicit_test_replacement_claim', 'explicit_check_to_test_claim'}:
                lines.append(f"- 显式替代 {link['old_id']} → {link['new_id']}：{link['reason']}（覆盖关系为模型声明，不是程序证明）")
    for risk in report.get('risks', []):
        lines.append(f'- 审阅提醒（不自动判定未完成）：{risk}')
    return '\n'.join(lines)
