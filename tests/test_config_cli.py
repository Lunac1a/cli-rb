import pytest
from typer.testing import CliRunner

from cli_rb.cli import app
from cli_rb.config import ModelConfig


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "http://localhost.evil/v1",
    "http://key@localhost/v1", "http://localhost/v1?key=secret", "file:///tmp/model",
])
def test_no_accidental_cloud_or_embedded_credentials(url):
    with pytest.raises(ValueError):
        ModelConfig(base_url=url)


def test_demo_is_explicit_and_disables_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    result = CliRunner().invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert "模拟模型演示" in result.output
    assert "模型请求工具" in result.output
    assert "工具返回" in result.output
    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_smoke_does_not_fallback_without_model(monkeypatch):
    monkeypatch.delenv("CLI_RB_MODEL", raising=False)
    result = CliRunner().invoke(app, ["smoke"])
    assert result.exit_code == 1
    assert "没有启用替代模型" in result.output
