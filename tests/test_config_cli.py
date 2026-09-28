import pytest
from typer.testing import CliRunner

from cli_rb.interface.cli import app
from cli_rb.config import ModelConfig


def test_local_output_default_and_slow_request_payload(monkeypatch):
    for key in ("CLI_RB_ENV_FILE", "CLI_RB_MAX_OUTPUT_TOKENS", "CLI_RB_REQUEST_TIMEOUT_SECONDS", "CLI_RB_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLI_RB_MODEL", "test-local")
    monkeypatch.setenv("CLI_RB_BASE_URL", "http://localhost:12345/v1")
    assert ModelConfig().max_output_tokens == ModelConfig.from_env().max_output_tokens == 32768
    monkeypatch.setenv("CLI_RB_REQUEST_TIMEOUT_SECONDS", "600")
    model = ModelConfig.from_env().build_model()
    payload = model._get_request_payload("hello")
    assert payload.get("max_completion_tokens", payload.get("max_tokens")) == 32768
    assert model.request_timeout == model.stream_chunk_timeout == 600
    monkeypatch.setenv("CLI_RB_MAX_OUTPUT_TOKENS", "8192")
    assert ModelConfig.from_env().max_output_tokens == 8192


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


def test_explicit_api_file_and_secret_not_serialized(tmp_path, monkeypatch):
    from dataclasses import asdict
    for name in ("CLI_RB_BASE_URL", "CLI_RB_MODEL", "CLI_RB_API_KEY", "CLI_RB_ALLOW_REMOTE"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "api.env"
    path.write_text('CLI_RB_BASE_URL="https://example.com/v1"\nCLI_RB_MODEL="test"\n'
                    'CLI_RB_API_KEY="private-test-value"\nCLI_RB_ALLOW_REMOTE="1"\n')
    monkeypatch.setenv("CLI_RB_ENV_FILE", str(path))
    config = ModelConfig.from_env()
    assert config.api_key() == "private-test-value"
    assert "private-test-value" not in str(asdict(config))
    assert config.build_model().openai_api_base == "https://example.com/v1"


def test_remote_requires_key_and_https(monkeypatch):
    monkeypatch.delenv("CLI_RB_ENV_FILE", raising=False)
    monkeypatch.delenv("CLI_RB_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ModelConfig("http://example.com/v1", "test", allow_remote=True)
    with pytest.raises(ValueError, match="CLI_RB_API_KEY"):
        ModelConfig("https://example.com/v1", "test", allow_remote=True).build_model()
