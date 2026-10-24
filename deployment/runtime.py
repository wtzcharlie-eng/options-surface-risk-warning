"""Validate platform settings and bridge MODEL_* into the existing assistant.

No keys are serialized. A blank platform endpoint leaves the app usable in the
explicitly labelled local-evidence mode; it never proves platform LLM readiness.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlsplit
import os
import re


class DeploymentConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    base_path: str = ""
    model_provider: str = "platform-qwen"
    model_name: str = "qwen3.8-27b-vllm"
    model_base_url: str = ""
    model_api_key: str = field(default="", repr=False)
    model_timeout_seconds: float = 90.
    model_max_tokens: int = 1400
    enable_thinking: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None):
        env = os.environ if env is None else env
        try:
            port = int(env.get("SERVICE_PORT", "8000"))
            timeout = float(env.get("MODEL_TIMEOUT_SECONDS", "90"))
            max_tokens = int(env.get("MODEL_MAX_TOKENS", "1400"))
        except (ValueError, TypeError):
            raise DeploymentConfigError("SERVICE_PORT 和 MODEL_TIMEOUT_SECONDS 必须为有效数字。") from None
        thinking = env.get("MODEL_ENABLE_THINKING", "false").lower().strip()
        if thinking not in {"true", "false"}:
            raise DeploymentConfigError("MODEL_ENABLE_THINKING 仅支持 true 或 false。")
        obj = cls(host=env.get("SERVICE_HOST", "0.0.0.0").strip(), port=port,
                  base_path=env.get("SERVICE_BASE_PATH", "").strip().strip("/"),
                  model_provider=env.get("MODEL_PROVIDER", "platform-qwen").strip(),
                  model_name=env.get("MODEL_NAME", "qwen3.8-27b-vllm").strip(),
                  model_base_url=env.get("MODEL_BASE_URL", "").strip(),
                  model_api_key=env.get("MODEL_API_KEY", "").strip(),
                  model_timeout_seconds=timeout, model_max_tokens=max_tokens, enable_thinking=thinking == "true")
        obj.validate()
        return obj

    def validate(self):
        if not 1 <= self.port <= 65535:
            raise DeploymentConfigError("SERVICE_PORT 必须位于 1–65535。")
        if self.host not in {"0.0.0.0", "127.0.0.1", "::"}:
            raise DeploymentConfigError("SERVICE_HOST 仅支持 0.0.0.0、127.0.0.1 或 ::。")
        if self.base_path and not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", self.base_path):
            raise DeploymentConfigError("SERVICE_BASE_PATH 必须为相对 URL 路径。")
        if self.model_provider not in {"platform-qwen", "domestic-compatible"}:
            raise DeploymentConfigError("比赛配置请选择 platform-qwen 或 domestic-compatible。")
        if not 1 <= self.model_timeout_seconds <= 120:
            raise DeploymentConfigError("MODEL_TIMEOUT_SECONDS 必须位于 1–120 秒。")
        if not 64 <= self.model_max_tokens <= 4096:
            raise DeploymentConfigError("MODEL_MAX_TOKENS 必须位于 64–4096。")
        if self.model_base_url:
            parsed = urlsplit(self.model_base_url)
            blocked = {"openai.com", "anthropic.com", "googleapis.com", "google.com"}
            host = (parsed.hostname or "").lower()
            if any(host == x or host.endswith("." + x) for x in blocked):
                raise DeploymentConfigError("该服务不属于比赛支持的平台本地或国产模型配置。")
            # Reuse the existing transport's URL and timeout checks. Missing
            # credentials are tracked separately as pending configuration.
            from vol_surface.risk_assistant import LLMConfig, AssistantConfigError
            try:
                LLMConfig(provider="openai-compatible", model=self.model_name or "pending-model",
                          base_url=self.model_base_url, api_key=self.model_api_key or "format-check-only",
                          timeout_seconds=self.model_timeout_seconds).validate()
            except AssistantConfigError:
                raise DeploymentConfigError("MODEL_BASE_URL、模型名称或密钥格式无效；请核对平台的 HTTPS 兼容接口配置。") from None

    @property
    def model_ready(self):
        return bool(self.model_base_url and self.model_name and self.model_api_key)

    def apply(self):
        # In competition mode an unrelated OPENAI_* environment cannot enable
        # a foreign provider or supply its credential to the platform endpoint.
        os.environ.update({"RISK_DEPLOYMENT_PROFILE": "competition",
                           "RISK_LLM_PROVIDER": "openai-compatible" if self.model_ready else "local",
                           "RISK_LLM_BASE_URL": self.model_base_url,
                           "RISK_LLM_MODEL": self.model_name,
                           "RISK_LLM_API_KEY": self.model_api_key,
                           "RISK_LLM_TIMEOUT_SECONDS": str(self.model_timeout_seconds),
                           "RISK_LLM_MAX_TOKENS": str(self.model_max_tokens),
                           "RISK_LLM_COMPACT_EVIDENCE": "true",
                           "RISK_LLM_ENABLE_THINKING": str(self.enable_thinking).lower()})

    def public_status(self):
        return {"host": self.host, "port": self.port, "base_path": self.base_path,
                "model_provider": self.model_provider, "model_configured": self.model_ready,
                "missing_model_fields": [name for name, value in
                    (("MODEL_BASE_URL", self.model_base_url), ("MODEL_NAME", self.model_name),
                     ("MODEL_API_KEY", self.model_api_key)) if not value],
                "remote_call_verified": False}


def load_settings(root: Path):
    from dotenv import load_dotenv
    load_dotenv(root / ".env", override=False)
    return RuntimeSettings.from_env()
