from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


BackendRoot = Path(__file__).resolve().parents[2]
ProjectRoot = BackendRoot.parent
RepositoryRoot = ProjectRoot.parents[1]


def _LoadEnvFiles() -> None:
    for EnvPath in (
        RepositoryRoot / ".env",
        ProjectRoot / ".env",
        BackendRoot / ".env",
    ):
        if EnvPath.exists():
            load_dotenv(EnvPath, override=False)


def _GetInt(Name: str, Default: int) -> int:
    RawValue = os.getenv(Name)
    if RawValue is None or RawValue.strip() == "":
        return Default
    return int(RawValue)


def _GetUrl(Name: str, Default: str) -> str:
    return os.getenv(Name, Default).rstrip("/")


@dataclass(frozen=True)
class GatewaySettings:
    Host: str = field(default_factory=lambda: os.getenv("GATEWAY_HOST", "0.0.0.0"))
    Port: int = field(default_factory=lambda: _GetInt("GATEWAY_PORT", 8002))
    AsrBaseUrl: str = field(default_factory=lambda: _GetUrl("ASR_BASE_URL", "http://127.0.0.1:8006"))
    TtsBaseUrl: str = field(default_factory=lambda: _GetUrl("TTS_BASE_URL", "http://127.0.0.1:8004"))
    CorsAllowOrigins: Tuple[str, ...] = (
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )


@dataclass(frozen=True)
class AsrSettings:
    Host: str = field(default_factory=lambda: os.getenv("ASR_HOST", "0.0.0.0"))
    Port: int = field(default_factory=lambda: _GetInt("ASR_PORT", 8006))
    DashScopeApiKey: str | None = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_API_KEY")
    )
    PrimaryModel: str = field(default_factory=lambda: os.getenv("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash"))
    FallbackModel: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_ASR_FALLBACK_MODEL", "paraformer-realtime-v1")
    )


@dataclass(frozen=True)
class TtsSettings:
    Host: str = field(default_factory=lambda: os.getenv("TTS_HOST", "0.0.0.0"))
    Port: int = field(default_factory=lambda: _GetInt("TTS_PORT", 8004))


@dataclass(frozen=True)
class ChatSettings:
    Host: str = field(default_factory=lambda: os.getenv("CHAT_HOST", "0.0.0.0"))
    Port: int = field(default_factory=lambda: _GetInt("CHAT_PORT", 8003))
    BailianAppId: str = field(
        default_factory=lambda: os.getenv("BAILIAN_APP_ID", "4dc0700043fc46679e1568339e580678")
    )


@dataclass(frozen=True)
class AppSettings:
    Gateway: GatewaySettings = field(default_factory=GatewaySettings)
    Asr: AsrSettings = field(default_factory=AsrSettings)
    Tts: TtsSettings = field(default_factory=TtsSettings)
    Chat: ChatSettings = field(default_factory=ChatSettings)


def LoadAppSettings() -> AppSettings:
    _LoadEnvFiles()
    return AppSettings()
