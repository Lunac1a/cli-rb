"""Local checkpoints plus a conservative application-level resume gate."""

import json
import hashlib
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END


class SessionNotReady(RuntimeError):
    pass


def read_session_record(path: Path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def inspect_session(directory: Path):
    """Read recorded evidence only; never open a graph or authorize resumption."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('会话目录不存在。')

    conversation = read_session_record(directory / 'conversation.json')
    states = {'idle': '记录为完整结束', 'running': '记录为运行中，可能已中断',
              'blocked': '上轮异常结束，已阻止恢复'}
    lines = [f'会话检查：{directory}',
             states.get(str(conversation.get('state')), '没有有效的会话状态记录'),
             '以下仅为磁盘记录；未检查当前文件内容、进程或检查点是否可恢复。',
             '仅列出已有动作日志，按记录名排列，不代表完整调用清单或执行顺序。']
    journals = sorted((directory / 'tool-logs').glob('*-action.json'))
    returned = 0
    for path in journals:
        record = read_session_record(path)
        receipt = path.with_name(path.name.removesuffix('-action.json') + '.json')
        result = read_session_record(receipt).get('result')
        label = '结果不确定；可能已有修改，不得据此重跑'
        if record.get('state') == 'returned' and isinstance(result, str):
            returned += 1
            label = '已返回；不代表成功'
            if result.startswith('工具错误'):
                label = '已返回工具错误'
            elif record.get('action') == 'command':
                try:
                    outcome = json.loads(result)
                except ValueError:
                    outcome = None
                if isinstance(outcome, dict):
                    label = (f"命令已返回：退出码 {outcome.get('exit_code', '未知')}，"
                             f"超时 {outcome.get('timed_out', '未知')}")
        lines.append(f"{record.get('action', '未知工具')}：{label}\n  记录：{path}")
        if receipt.is_file():
            lines.append(f'  结果日志：{receipt}')
    lines.append(f'已返回记录 {returned}；结果不确定记录 {len(journals) - returned}。')
    if not journals:
        lines.append('没有工具动作记录；不证明工作区从未被修改。')
    lines.append('这是程序证据说明，没有调用模型、执行工具或改写会话状态；不代表任务完成。')
    return '\n'.join(lines)


def recovery_messages(messages):
    """Close missing tool results truthfully, only in the new checkpoint lineage."""
    result, missing = [], []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            raise SessionNotReady('历史工具消息配对异常，拒绝自动整理。')
        result.append(message)
        index += 1
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        calls = {call['id']: call for call in message.tool_calls}
        if len(calls) != len(message.tool_calls):
            raise SessionNotReady('历史工具调用ID重复，拒绝自动整理。')
        replies = {}
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            reply = messages[index]
            if reply.tool_call_id not in calls or reply.tool_call_id in replies:
                raise SessionNotReady('历史工具消息配对异常，拒绝自动整理。')
            replies[reply.tool_call_id] = reply
            index += 1
        for call in message.tool_calls:
            reply = replies.get(call['id'])
            if reply is None:
                missing.append(call)
                reply = ToolMessage(
                    content='程序恢复记录：上轮中断，此请求没有已确认的返回结果。可能已执行并产生修改；没有重放，不得推断未执行或成功。',
                    tool_call_id=call['id'], name=call['name'], status='error',
                )
            result.append(reply)
    result.append(AIMessage(
        content='程序恢复记录：用户检查了中断记录和当前差异，同意保留现状继续。旧轮次已停止，结果不确定的动作仍不确定。只处理下一条新要求，不根据旧工具请求重跑；先读取当前文件再决定后续步骤，写操作必须重新获得授权。',
        response_metadata={'source': 'interruption_recovery'},
    ))
    return result, missing


class SessionStore:
    """Hold an OS-released SQLite lock for the whole conversation process."""

    def __init__(self, directory: Path, *, create=False, model_config=None, allow_interrupted=False):
        self.directory = directory
        self.path = directory / 'conversation.json'
        self.create = create
        self.model_config = model_config
        self.allow_interrupted = allow_interrupted
        self.lock = None

    def __enter__(self):
        self.lock = sqlite3.connect(self.directory / 'conversation-lock.sqlite', timeout=0)
        try:
            try:
                self.lock.execute('BEGIN EXCLUSIVE')
            except sqlite3.OperationalError as error:
                raise SessionNotReady('此会话正被其他进程占用或无法锁定，未开始执行。') from error
            if self.create:
                if self.path.exists():
                    raise SessionNotReady('会话记录已存在，不能覆盖。')
                self.data = {'version': 1, 'state': 'idle', 'checkpoint_id': None,
                             'model_config': self.model_config}
                self.save()
            else:
                if not self.path.is_file() or not (self.directory / 'checkpoints.sqlite').is_file():
                    raise SessionNotReady('此会话没有可恢复的对话检查点；旧会话不会自动迁移。')
                self.data = json.loads(self.path.read_text(encoding='utf-8'))
                if self.data.get('version') != 1 or self.data.get('state') not in {'idle', 'running', 'blocked'}:
                    raise SessionNotReady('会话状态格式不支持恢复。')
                if not self.allow_interrupted:
                    self.require_idle()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.lock:
            self.lock.close()
            self.lock = None

    @property
    def config(self):
        return {'configurable': {'thread_id': self.data.get('thread_id', self.directory.name)}}

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def require_idle(self):
        if self.data.get('version') != 1 or self.data.get('state') != 'idle':
            raise SessionNotReady('上轮未完整结束，当前不能恢复执行；修改和日志保留，没有重放工具。')
        fingerprints = self.action_fingerprints()
        reviewed = self.data.get('reviewed_actions', {})
        if any(fingerprints.get(name) != digest for name, digest in reviewed.items()):
            raise SessionNotReady('已核对的旧工具记录发生变化，需重新检查后才能继续。')
        for name in fingerprints:
            if name in reviewed:
                continue  # Acknowledged uncertainty is not rewritten as success.
            path = self.directory / 'tool-logs' / name
            receipt = path.with_name(path.name.removesuffix('-action.json') + '.json')
            if (read_session_record(path).get('state') != 'returned'
                    or not isinstance(read_session_record(receipt).get('result'), str)):
                raise SessionNotReady('发现结果不确定或被中断的工具记录，拒绝恢复执行；没有重放工具。')

    def action_fingerprints(self):
        result = {}
        for path in sorted((self.directory / 'tool-logs').glob('*-action.json')):
            receipt = path.with_name(path.name.removesuffix('-action.json') + '.json')
            digest = hashlib.sha256(path.read_bytes())
            digest.update(b'\0receipt\0')
            digest.update(receipt.read_bytes() if receipt.is_file() else b'\0missing')
            result[path.name] = digest.hexdigest()
        return result

    async def recover(self, agent, reviewed_state, fingerprints):
        """Called only after CLI review/confirmation and container quiescence checks."""
        current = await agent.graph.aget_state(self.config)
        if current.config != reviewed_state.config or self.action_fingerprints() != fingerprints:
            raise SessionNotReady('核对期间检查点或工具记录改变，未切换会话，请重新检查。')
        messages, missing = recovery_messages(list(current.values.get('messages', [])))
        old_config = self.config
        next_thread = self.directory.name + '-continue-' + uuid4().hex
        next_config = {'configurable': {'thread_id': next_thread}}
        # Write state only. START seeds messages; END clears scheduled tasks without execution.
        await agent.graph.aupdate_state(next_config, {'messages': messages}, as_node=START)
        await agent.graph.aupdate_state(next_config, None, as_node=END)
        ready = await agent.graph.aget_state(next_config)
        if ready.next or ready.tasks or len(ready.values.get('messages', [])) != len(messages):
            raise SessionNotReady('无法建立无待执行动作的继续记录；原会话未切换。')
        if self.action_fingerprints() != fingerprints:
            raise SessionNotReady('保存期间工具记录改变；原会话未切换，请重新检查。')
        recovery = {'from_config': current.config or old_config, 'to_thread': next_thread,
                    'missing_tool_ids': [call['id'] for call in missing],
                    'reviewed_actions': fingerprints}
        previous = self.data
        self.data = {**previous, 'thread_id': next_thread, 'state': 'idle',
                     'checkpoint_id': ready.config['configurable']['checkpoint_id'],
                     'reviewed_actions': fingerprints, 'review_writes': True,
                     'recoveries': [*previous.get('recoveries', []), recovery]}
        try:
            self.save()  # Atomic pointer switch; an earlier crash leaves old blocked state active.
        except BaseException:
            self.data = previous
            raise
        return list(ready.values['messages'])

    async def history(self, agent):
        self.require_idle()
        state = await agent.graph.aget_state(self.config)
        checkpoint_id = state.config.get('configurable', {}).get('checkpoint_id')
        if state.next or state.tasks or checkpoint_id != self.data['checkpoint_id']:
            raise SessionNotReady('检查点仍有待执行节点或与完成记录不一致，拒绝恢复执行。')
        return list(state.values.get('messages', []))

    async def begin(self, agent):
        history = await self.history(agent)
        self.data['state'] = 'running'
        self.save()  # Persist before any model or tool invocation.
        return history

    async def finish(self, agent):
        from cli_rb.agent import has_reply
        state = await agent.graph.aget_state(self.config)
        messages = state.values.get('messages', [])
        if state.next or state.tasks or not messages or not has_reply(messages[-1]):
            raise SessionNotReady('本轮没有形成完整最终答复，记录保留，但暂不允许恢复执行。')
        self.data.update(state='idle', checkpoint_id=state.config['configurable']['checkpoint_id'])
        self.save()

    def block(self):
        self.data['state'] = 'blocked'
        self.save()
