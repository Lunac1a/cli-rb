import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from cli_rb.interface.presentation import progress_entries
from cli_rb.quality.test_evidence import parse_test_evidence


def test_progress_preserves_observed_obstacle_and_hides_internal_content():
    messages = [HumanMessage(content='task'), AIMessage(content='PRIVATE', tool_calls=[
        {'name': 'run_command', 'args': {'command': 'PRIVATE'}, 'id': '1'}]),
        ToolMessage(content='{"exit_code":1,"output":"PRIVATE"}', tool_call_id='1'),
        AIMessage(content='', tool_calls=[
            {'name': 'run_verification', 'args': {}, 'id': '2'}]),
        ToolMessage(content='{"exit_code":0}', tool_call_id='2')]
    text = '\n'.join(progress_entries(messages))
    assert '命令未成功' in text and '正在运行验证' in text and '不代表测试全部通过' in text
    assert 'PRIVATE' not in text
    assert progress_entries(messages + [HumanMessage(content='new task')]) == []


def test_stdlib_checks_work_without_site_packages_and_detect_real_failure():
    # -S excludes installed pytest and all site-packages; no dependency installation.
    for expected, outcome in [(2, 'passed'), (3, 'failed')]:
        script = ('import unittest\n'
                  'class Contract(unittest.TestCase):\n'
                  f' def test_sum(self): self.assertEqual(1 + 1, {expected})\n'
                  'unittest.main(verbosity=2)\n')
        run = subprocess.run([sys.executable, '-S', '-c', script], capture_output=True,
                             text=True, timeout=10)
        parsed = parse_test_evidence({'output': run.stdout + run.stderr, 'exit_code': run.returncode})
        assert parsed['framework'] == 'unittest'
        assert parsed['counts'][outcome] == 1 and parsed['total'] == 1
        assert (run.returncode == 0) == (outcome == 'passed')
