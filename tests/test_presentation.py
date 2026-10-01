from io import StringIO
from rich.console import Console
from cli_rb.interface.presentation import print_answer, activity
from cli_rb.interface.cli import render
from langchain_core.messages import AIMessage, ToolMessage


def test_tool_payload_and_intermediate_prose_are_hidden(monkeypatch):
    output = StringIO()
    monkeypatch.setattr('cli_rb.interface.cli.console', Console(file=output, width=100))
    render(AIMessage(content='PRIVATE_INTERMEDIATE', tool_calls=[
        {'name':'read_file', 'args':{'path':'app.py', 'extra':'PRIVATE_ARGUMENT'}, 'id':'1'}]))
    render(ToolMessage(content='PRIVATE_FILE_CONTENT', tool_call_id='1', name='read_file'))
    assert '正在阅读文件' in output.getvalue()
    assert 'PRIVATE_' not in output.getvalue()


def test_markdown_answer_and_compact_evidence():
    output = StringIO()
    print_answer(Console(file=output, width=100),
                 '**完成**\n\n```python\nprint(42)\n```\n[程序核对] 部分完成。详细原因\n- receipt-private-id')
    text = output.getvalue()
    assert 'print(42)' in text and '**完成**' not in text
    assert '/details' in text and '部分完成' in text
    assert 'receipt-private-id' not in text


def test_status_does_not_echo_command():
    assert activity({'name':'run_command','args':{'command':'secret'}}) == '正在执行命令…'


def test_details_hide_internal_audit_but_preserve_failed_and_unknown_checks():
    from cli_rb.interface.presentation import print_details
    output = StringIO()
    text = ('[完成状态] 内部状态\n\n使用 `python todo.py list`。缺少 pytest，测试未完成。'
            '\n\n[程序核对] 部分完成。verification-secret: internal failure')
    report = {'status': 'needs_attention', 'evidence': [
        {'id': 'verification-secret', 'remaining_outcome': 'failed', 'current': True},
        {'remaining_outcome': 'unknown', 'current': True},
        {'remaining_outcome': 'passed', 'current': False}]}
    print_details(Console(file=output, width=120), text, report)
    result = output.getvalue()
    assert 'python todo.py list' in result and '缺少 pytest' in result
    assert '未通过' in result and '结果未能确认' in result and '修改前' in result
    assert 'verification-secret' not in result and '[完成状态]' not in result
    assert '内部状态' not in result and 'internal failure' not in result


def test_inspection_commands_do_not_call_model(monkeypatch):
    from types import SimpleNamespace
    from cli_rb.interface.cli import chat_loop
    output = StringIO()
    monkeypatch.setattr('cli_rb.interface.cli.console', Console(file=output, width=100))
    inputs = iter(['/status', '/changes', '/details', '/exit'])
    backend = SimpleNamespace(snapshot=SimpleNamespace(workspace='project', session='session'),
                              patches=SimpleNamespace(diff=lambda: '+added'), cancel_all=lambda: None)
    chat_loop(SimpleNamespace(prompt=lambda _: next(inputs)), None, backend,
              [AIMessage(content='**saved answer**')], None)
    assert 'project' in output.getvalue() and 'added' in output.getvalue()
    assert 'saved answer' in output.getvalue()
