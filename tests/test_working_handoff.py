import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from helpers import ScriptedModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END

from cli_rb.interface.demo import demo_backend
from cli_rb.runtime.agent import astream_turn, build_agent
from cli_rb.runtime.context import ContextPolicy
from cli_rb.runtime.history import HistoryArchive
from cli_rb.runtime.persistence import SessionStore


@pytest.fixture
def folder():
    path = Path('.cli-rb/handoff-tests') / uuid4().hex
    path.mkdir(parents=True)
    return path


def test_total_target_counts_pinned_users_summary_and_schemas(folder):
    async def run():
        with SessionStore(folder, create=True) as store:
            # More than the old 500-character limit, yet within the new token budget.
            handoff = 'Verified file path and outcome; unresolved next step. ' * 20
            model = ScriptedModel(replies=[AIMessage(content=handoff)])
            context = build_agent(model, demo_backend(), session=store,
                                  context_policy=ContextPolicy(window=32768)).context
            messages = []
            for i in range(12):
                messages += [HumanMessage(content=f'Constraint {i}: ' + 'u' * 250),
                             AIMessage(content='a' * 3600)]
            messages.append(HumanMessage(content='Continue.'))
            update = await context.abefore_model({'messages': messages}, None)
            assert update['context_summary_count'] == 1
            assert update['context_estimate'] <= int(context.policy.input_limit * .20)
            for m in messages:
                if isinstance(m, HumanMessage):
                    assert m in update['messages']
            assert handoff.strip() in update['messages'][1].text
            assert 'Constraint 0:' in model.seen[0][-1].text
            assert 'at most 500 characters' not in model.seen[0][-1].text
    asyncio.run(run())


def test_graph_recovers_exact_archived_text_and_archive_survives_reopen(folder):
    async def run():
        marker = 'EXACT historical line: beta / 731; not verified.'
        old = 'x' * 40000 + marker + 'y' * 8000
        def call(args, cid):
            return AIMessage(content='', tool_calls=[{'name': 'read_history', 'args': args, 'id': cid}])
        model = ScriptedModel(replies=[
            AIMessage(content='Older investigation did not verify success. Consult archived messages.'),
            call({'query': 'EXACT historical'}, 'search'),
            call({'message_id': 'original', 'start_char': 40000, 'max_chars': len(marker)}, 'read'),
            AIMessage(content=marker),
        ])
        policy = ContextPolicy(window=32768)
        with SessionStore(folder, create=True) as store:
            async with AsyncSqliteSaver.from_conn_string(str(folder/'checkpoints.sqlite')) as saver:
                agent = build_agent(model, demo_backend(), session=store, checkpointer=saver,
                                    context_policy=policy, model_call_limit=4)
                await agent.graph.aupdate_state(store.config, {'messages': [
                    HumanMessage(content='Keep the old exact line available.', id='u1'),
                    AIMessage(content=old, id='original')]}, as_node=START)
                await agent.graph.aupdate_state(store.config, None, as_node=END)
                await store.finish(agent)
                events = [e async for e in astream_turn(agent, [HumanMessage(content='Find the exact historical line.')])]
                assert events[-1].text == marker
                result = next(m for m in model.seen[-1] if isinstance(m, ToolMessage) and m.tool_call_id == 'read')
                page = json.loads(result.text)
                assert page['text'] == marker and page['next_char'] == 40000 + len(marker)
                state = await agent.graph.aget_state(store.config)
                assert state.values['thread_model_call_count'] == 4
                assert not any(m.content == old for m in state.values['messages'])
        with SessionStore(folder):
            archive = HistoryArchive(folder)
            assert archive.read('EXACT historical')['messages'][0]['id'] == 'original'
            assert archive.read(message_id='original', start_char=40000, max_chars=len(marker))['text'] == marker
    asyncio.run(run())


def test_repeated_compaction_preserves_archive_and_accepts_soft_target_overrun(folder):
    async def run():
        with SessionStore(folder, create=True) as store:
            model = ScriptedModel(replies=[AIMessage(content='handoff one'), AIMessage(content='z' * 6000)])
            context = build_agent(model, demo_backend(), session=store,
                                  context_policy=ContextPolicy(window=32768)).context
            first = [AIMessage(content='original needle '+ 'a' * 48000, id='old'), HumanMessage(content='continue', id='u')]
            update = await context.abefore_model({'messages': first}, None)
            second = [*update['messages'][1:], AIMessage(content='b' * 48000), HumanMessage(content='next', id='new')]
            rejected = await context.abefore_model({'messages': second}, None)
            assert 'messages' in rejected
            assert HistoryArchive(folder).read('original needle')['messages'][0]['id'] == 'old'
            records = [json.loads(line) for line in (folder/'context-calls.jsonl').read_text().splitlines()]
            assert records[-1]['outcome'] == 'accepted'
            assert records[-1]['above_soft_target'] is True
            assert 'handoff one' in model.seen[-1][-1].text
    asyncio.run(run())


def test_archive_pages_long_tool_arguments_and_excludes_hidden_reasoning(folder):
    archive = HistoryArchive(folder)
    message = AIMessage(content=[{'type': 'text', 'text': 'Visible.'},
                                 {'type': 'reasoning', 'encrypted_content': 'PRIVATE'}],
                        tool_calls=[{'name': 'edit_file', 'args': {'text': 'a' * 20000}, 'id': 'tool'}],
                        id='call')
    archive.append([message])
    archive.append([message])
    assert len(archive.read()['messages']) == 1
    assert archive.read('PRIVATE')['messages'] == []
    page = archive.read(message_id='call', max_chars=1000)
    assert len(page['text']) == 1000 and page['next_char'] == 1000
    with pytest.raises(ValueError):
        archive.read(message_id='call', start_char=-1)
