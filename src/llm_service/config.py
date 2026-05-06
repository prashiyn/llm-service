"""Settings and config directory resolution for llm-service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "llm-service"
    debug: bool = False
    config_dir: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    service_auth_token: str | None = Field(default=None, validation_alias="SERVICE_AUTH_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_config_dir() -> Path:
    s = get_settings()
    if s.config_dir:
        return Path(s.config_dir).resolve()
    cwd = Path.cwd()
    for candidate in (cwd / "src" / "config", cwd / "config"):
        if candidate.is_dir():
            return candidate.resolve()
    return (cwd / "config").resolve()
