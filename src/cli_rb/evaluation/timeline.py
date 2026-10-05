"""Offline reconstruction of observable work, never hidden model reasoning.

Usage: python -m cli_rb.evaluation.timeline SOLVER_DIR --output NEW_DIRECTORY
Only read trusted local CLI Rb checkpoints. No graph, tools, or model is executed.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def public(value):
    if isinstance(value, dict):
        if value.get('type') in {'reasoning', 'thinking', 'redacted_thinking'}:
            return None
        return {k: public(v) for k, v in value.items()
                if k not in {'encrypted_content', 'reasoning_content', 'additional_kwargs'}}
    if isinstance(value, list):
        return [clean for v in value if (clean := public(v)) is not None]
    return value


def text(content):
    if isinstance(content, str):
        return content
    return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict)
                     and b.get('type') == 'text')


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()] if path.exists() else []


def reconstruct(solver):
    solver = Path(solver).resolve()
    sessions = sorted((solver / 'sessions').glob('*/checkpoints.sqlite'))
    if len(sessions) != 1:
        raise ValueError('Expected exactly one checkpoint session; select a single solver attempt.')
    db = sessions[0]
    sources = {}

    def source(p):
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return str(p)

    source(db)
    steps, observed, calls, last_versions = [], set(), {}, {}
    serde = JsonPlusSerializer()
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as conn:
        rows = conn.execute('SELECT thread_id,checkpoint_ns,checkpoint_id,type,checkpoint '
                            'FROM checkpoints ORDER BY checkpoint_id').fetchall()
    if len({(r[0], r[1]) for r in rows}) != 1:
        raise ValueError('Multiple checkpoint lineages/namespaces require separate analysis.')
    for _, _, checkpoint_id, kind, blob in rows:
        checkpoint = serde.loads_typed((kind, blob))
        messages = checkpoint.get('channel_values', {}).get('messages', [])
        context_ids = []
        for position, m in enumerate(messages):
            mid = m.id or f'position-{position}'
            data = {'message_id': mid, 'role': m.type, 'text': text(m.content)}
            if m.type == 'ai':
                data['tool_calls'] = public(m.tool_calls)
            if m.type == 'tool':
                data.update(tool_call_id=m.tool_call_id, tool_name=m.name,
                            status=getattr(m, 'status', None))
            signature = json.dumps(data, ensure_ascii=False, sort_keys=True)
            context_ids.append(mid)
            if signature in observed:
                continue
            observed.add(signature)
            number = len(steps) + 1
            step = {'step': number, **data, 'source': str(db), 'checkpoint_id': checkpoint_id,
                    'saved_context_message_ids': list(context_ids),
                    'revises_step': last_versions.get(mid), 'elapsed_seconds': None}
            last_versions[mid] = number
            if m.type == 'human':
                step['phase'] = ('program_feedback' if data['text'].startswith('Program completion check')
                                 else 'request_or_context')
            elif m.type == 'ai':
                step['phase'] = 'model_tool_selection' if m.tool_calls else 'model_reply_or_draft'
                for call in m.tool_calls:
                    calls.setdefault(call['id'], number)
            elif m.type == 'tool':
                step['phase'] = 'tool_result'
                step['responds_to_step'] = calls.get(m.tool_call_id)
            else:
                step['phase'] = 'context_message'
            steps.append(step)

    events_path = solver / 'events.jsonl'
    events = read_lines(events_path)
    if events_path.exists():
        source(events_path)
    for step in steps:
        matches = [(i, x) for i, x in enumerate(events, 1)
                   if x.get('event', {}).get('data', {}).get('id') == step['message_id']
                   and text(x['event']['data'].get('content', '')) == step['text']]
        if matches:
            i, event = matches[0]
            step.update(elapsed_seconds=event.get('elapsed'), event_line=i)

    ledgers = {}
    for name in ['verification', 'verification-plan', 'completion-checks', 'stop-diagnostics',
                 'model-attempts', 'context-decisions', 'context-calls']:
        p = db.parent / (name + '.jsonl')
        if p.exists():
            source(p)
        ledgers[name] = [{'source': str(p), 'line': i, 'record': public(record)}
                         for i, record in enumerate(read_lines(p), 1)]
    for step in steps:
        step['verification_links'] = [x for x in ledgers['verification']
                                      if x['record'].get('id') and x['record']['id'] in step['text']]
    unresolved = [cid for cid in calls if not any(s.get('tool_call_id') == cid for s in steps)]
    result_path = solver / 'result.json'
    result = public(json.loads(result_path.read_text(encoding='utf-8'))) if result_path.exists() else None
    if result_path.exists():
        source(result_path)
    return {'schema_version': 1, 'solver': str(solver), 'steps': steps, 'ledgers': ledgers,
            'solver_result': result, 'source_sha256': sources,
            'summary': {'steps': len(steps), 'unique_tool_requests': len(calls),
                        'unmatched_tool_requests': unresolved,
                        'feedback_steps': [s['step'] for s in steps if s['phase'] == 'program_feedback']},
            'limits': ['顺序是检查点首次观察顺序；并行工具完成先后不一定可知。',
                       '保存的上下文不等于实际发送给provider的完整请求；系统提示、裁剪和最终输入未知。',
                       '只记录公开文字与行动，不恢复内部推理；工作假设仅以模型公开声明为依据。',
                       '独立日志保留各自顺序，不按行号猜测跨日志时序；缺失结果不证明未执行。',
                       '可见工具结果可能已截断；原始检查点和日志由来源路径定位。',
                       '不自动推断方向错误或能力不足，不把自述、工具返回或通过检查当作任务成功。']}


def render(report):
    esc = html.escape
    sections = []
    labels = {'request_or_context': '需求／上下文', 'program_feedback': 'Agent 程序反馈',
              'model_tool_selection': 'LLM 选择行动', 'model_reply_or_draft': 'LLM 回复／草稿',
              'tool_result': 'Agent 工具结果', 'context_message': '上下文消息'}
    for s in report['steps']:
        calls = ', '.join(c['name'] for c in s.get('tool_calls', []))
        title = calls or s['text'][:150] or s.get('tool_name') or '无公开文字'
        link = f" → 对应步骤 {s['responds_to_step']}" if s.get('responds_to_step') else ''
        sections.append(f'<details><summary>{s["step"]}. {labels[s["phase"]]}{link}：{esc(title)}</summary>'
                        f'<pre>{esc(json.dumps(s,ensure_ascii=False,indent=2))}</pre></details>')
    return ('<!doctype html><meta charset="utf-8"><title>任务工作时间线</title>'
            '<style>body{max-width:1100px;margin:35px auto;font:16px/1.6 system-ui;padding:0 20px}'
            'summary{cursor:pointer;padding:12px;background:#edf2f7;border-bottom:1px solid #ddd}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:8px 0}</style>'
            '<h1>任务工作时间线</h1><p>离线观察记录 · 未调用模型或重放工具</p><p>'
            + esc(report['solver']) + '</p><ul>'
            + ''.join('<li>'+esc(x)+'</li>' for x in report['limits'])+'</ul>'
            + '<h2>概况</h2><pre>'+esc(json.dumps(report['summary'],ensure_ascii=False,indent=2))+'</pre>'
            + ''.join(sections) + '<h2>验证、停止、预算与上下文记录（各日志独立顺序）</h2>'
            + ''.join('<details><summary>'+esc(k)+'</summary><pre>'
                      +esc(json.dumps(v,ensure_ascii=False,indent=2))+'</pre></details>'
                      for k,v in report['ledgers'].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solver', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = reconstruct(args.solver)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'timeline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'timeline.html').write_text(render(report),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__ == '__main__':
    main()
