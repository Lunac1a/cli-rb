import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from cli_rb.evaluation.timeline import reconstruct, render


def fixture_db(tmp_path, snapshots):
    solver = tmp_path/'solver'
    session = solver/'sessions'/'one'
    session.mkdir(parents=True)
    db = session/'checkpoints.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE checkpoints(thread_id, checkpoint_ns, checkpoint_id, type, checkpoint)')
        for i, messages in enumerate(snapshots):
            kind, blob = JsonPlusSerializer().dumps_typed({'channel_values': {'messages': messages}})
            conn.execute('INSERT INTO checkpoints VALUES(?,?,?,?,?)', ('one', '', str(i), kind, blob))
    return solver, session


def test_draft_feedback_revisions_and_pairing_are_preserved(tmp_path):
    request = HumanMessage(content='Fix both entry points', id='user')
    action = AIMessage(content=[{'type':'reasoning','encrypted_content':'SECRET'},
                               {'type':'text','text':'Inspect caller'}], id='action',
                       tool_calls=[{'name':'read_file','args':{'path':'a.py'},'id':'call1','type':'tool_call'}])
    result = ToolMessage(content='verification-1 <script>bad()</script>',tool_call_id='call1',id='result')
    draft = AIMessage(content='First draft',id='draft')
    feedback = HumanMessage(content='Program completion check: compare request',id='feedback')
    revised = AIMessage(content='Final answer',id='draft')
    solver, session = fixture_db(tmp_path, [[request,action], [request,action,result,draft],
                                          [request,action,result,draft,feedback],
                                          [request,action,result,revised,feedback]])
    (session/'verification.jsonl').write_text(json.dumps({'id':'verification-1','outcome':'passed'})+'\n')
    before = (session/'checkpoints.sqlite').read_bytes()
    report = reconstruct(solver)
    assert (session/'checkpoints.sqlite').read_bytes() == before
    assert report['summary']['unique_tool_requests'] == 1
    assert report['summary']['unmatched_tool_requests'] == []
    assert report['steps'][2]['responds_to_step'] == 2
    assert report['steps'][2]['verification_links'][0]['line'] == 1
    assert report['steps'][-1]['revises_step'] == 4
    assert report['summary']['feedback_steps'] == [5]
    assert 'SECRET' not in json.dumps(report)
    page = render(report)
    assert '<script>bad()' not in page and '&lt;script&gt;' in page


def test_missing_result_is_unknown_not_fabricated(tmp_path):
    action = AIMessage(content='',id='m',tool_calls=[{'name':'edit_file','args':{},'id':'lost','type':'tool_call'}])
    solver,_ = fixture_db(tmp_path,[[action]])
    report = reconstruct(solver)
    assert report['summary']['unmatched_tool_requests'] == ['lost']
    assert report['steps'][0]['elapsed_seconds'] is None
    assert report['solver_result'] is None


def test_ambiguous_sessions_are_rejected(tmp_path):
    solver,_ = fixture_db(tmp_path,[])
    (solver/'sessions'/'two').mkdir()
    (solver/'sessions'/'two'/'checkpoints.sqlite').touch()
    with pytest.raises(ValueError,match='exactly one'):
        reconstruct(solver)
