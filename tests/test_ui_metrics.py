from cli_rb.interface.presentation import SessionMetrics


def test_elapsed_freezes_at_end_and_restarts_next_turn(monkeypatch):
    now = [10.0]
    monkeypatch.setattr('cli_rb.interface.presentation.time.monotonic', lambda: now[0])
    metrics = SessionMetrics()
    assert '等待统计' in metrics.text()
    metrics.begin()
    metrics.estimate, metrics.limit = 2000, 10000
    now[0] = 75.0
    assert '01:05' in metrics.text() and '20%' in metrics.text()
    assert '估算' in metrics.text() and '输入预算' in metrics.text()
    metrics.finish(True)
    now[0] = 100.0
    assert '01:05' in metrics.text() and '已结束' in metrics.text()
    metrics.begin()
    assert '00:00' in metrics.text()
