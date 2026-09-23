"""Explicit local model configuration; never fall back to a paid provider."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = ""

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("第一块仅支持无嵌入凭据的本机模型地址，不会回退到云端。")

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.getenv("CLI_RB_BASE_URL", "http://localhost:1234/v1"),
            model=os.getenv("CLI_RB_MODEL", ""),
        )

    def build_model(self):
        if not self.model.strip():
            raise ValueError("请设置 CLI_RB_MODEL 为本地服务返回的模型 ID。")
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=os.getenv("CLI_RB_API_KEY", "lm-studio"),
            max_tokens=2048,
            timeout=120,
            max_retries=1,
            temperature=0,
            use_responses_api=False,
        )
