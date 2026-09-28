"""Explicit model configuration; remote APIs require opt-in, with no fallback."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


def model_settings():
    path = os.getenv("CLI_RB_ENV_FILE")
    values = dict(dotenv_values(path, interpolate=False, encoding="utf-8-sig")) if path else {}
    values.update({k: v for k, v in os.environ.items() if k.startswith("CLI_RB_")})
    return values


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = ""
    allow_remote: bool = False
    max_output_tokens: int = 32768
    request_timeout_seconds: float = 120
    use_responses_api: bool = False
    reasoning_effort: str | None = None

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or (not local and (not self.allow_remote or parsed.scheme != "https"))
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("远程模型需显式启用且使用HTTPS；地址不能嵌入凭据。")
        if self.max_output_tokens < 1:
            raise ValueError("模型输出预算必须大于0。")
        if self.request_timeout_seconds <= 0:
            raise ValueError("模型请求等待时间必须大于0。")

    @classmethod
    def from_env(cls):
        values = model_settings()
        return cls(
            base_url=values.get("CLI_RB_BASE_URL") or "http://localhost:1234/v1",
            model=values.get("CLI_RB_MODEL") or "",
            allow_remote=values.get("CLI_RB_ALLOW_REMOTE") == "1",
            max_output_tokens=int(values.get("CLI_RB_MAX_OUTPUT_TOKENS") or cls.max_output_tokens),
            request_timeout_seconds=float(values.get("CLI_RB_REQUEST_TIMEOUT_SECONDS") or cls.request_timeout_seconds),
            use_responses_api=values.get("CLI_RB_USE_RESPONSES_API") == "1",
            reasoning_effort=values.get("CLI_RB_REASONING_EFFORT") or None,
        )

    def api_key(self):
        key = model_settings().get("CLI_RB_API_KEY")
        if not key and urlparse(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("远程服务需要CLI_RB_API_KEY；请选择填写好的CLI_RB_ENV_FILE。")
        return key or "lm-studio"

    def build_model(self):
        if not self.model.strip():
            raise ValueError("请设置CLI_RB_MODEL为服务支持的模型ID。")
        protocol = (
            {
                "use_responses_api": True,
                "output_version": "responses/v1",
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": self.reasoning_effort} if self.reasoning_effort else None,
            }
            if self.use_responses_api
            else {"use_responses_api": False, "temperature": 0,
                  "reasoning_effort": self.reasoning_effort}
        )
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key(),
            max_tokens=self.max_output_tokens,
            timeout=self.request_timeout_seconds,
            stream_chunk_timeout=self.request_timeout_seconds,
            max_retries=1,
            **protocol,
        )
