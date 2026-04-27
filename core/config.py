"""
InBalance v6 — Application Settings
=====================================
All environment variables are declared here and validated at startup via
pydantic-settings. Import `settings` anywhere you need a config value.

Usage
-----
  from core.config import settings

  print(settings.model_version)
  print(settings.artifacts_dir)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INBALANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Cycle engine
    model_version: str = "v4"
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS — comma-separated list of allowed origins for the API.
    # Example: "https://app.inbalance.io,https://staging.inbalance.io"
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
