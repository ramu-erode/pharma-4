"""Runtime configuration. The only module that reads environment variables.

Values come from the process environment, then `.env`, then `.env.example` (defaults
live only there). In compose, every container gets `.env.example` and an optional
`.env` through `env_file`, and `SERVICE` selects which MQTT credentials apply.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service: str = "dev"

    # ISA-95 location of this POC (ADR-0001).
    site: str = "chennai"
    area: str = "upstream"
    line: str = "suite-1"

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "pharma"
    postgres_user: str = "pharma"
    postgres_password: str | None = None

    bolt_uri: str = "bolt://localhost:7687"
    neo4j_auth: str | None = None  # "user/password", same variable the neo4j image reads

    seed: int = 42

    # Time model (ADR-0009). All durations are simulated seconds unless named *_wall_s.
    sim_speed: float = 3600.0
    integration_step_s: float = 5.0
    publish_period_s: float = 60.0
    deadband_floor_s: float = 600.0
    heartbeat_wall_s: float = 30.0

    backfill_batches: int = Field(default=200, ge=1)

    models_dir: str = "models"  # trained models and their manifests (a volume in compose)

    # i3X read API (ADR-0016). An empty key turns authentication off (tests only).
    i3x_url: str = "http://localhost:8600/v1"
    i3x_port: int = 8600
    i3x_api_key: str | None = None

    # LLM assistant (ADR-0017). The key lives in .env only, never in .env.example.
    anthropic_api_key: str | None = None
    assistant_model: str = "claude-opus-5"

    def mqtt_credentials(self, service: str | None = None) -> tuple[str, str | None]:
        """Username and password for `service` (default: SERVICE from the environment).

        Explicit MQTT_USERNAME/MQTT_PASSWORD win; otherwise the service name is the
        username and MQTT_PASSWORD_<SERVICE> the password.
        """
        service = service or self.service
        username = self.mqtt_username or service
        password = self.mqtt_password
        if password is None:
            key = "MQTT_PASSWORD_" + service.upper().replace("-", "_")
            password = os.environ.get(key) or _dotenv_value(key)
        return username, password

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password or ''}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def neo4j_credentials(self) -> tuple[str, str]:
        user, _, password = (self.neo4j_auth or "neo4j/").partition("/")
        return user, password


def _dotenv_value(key: str) -> str | None:
    """Read one key from .env, falling back to .env.example (same order as Settings)."""
    for path in (".env", ".env.example"):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    name, sep, value = line.strip().partition("=")
                    if sep and name == key:
                        return value
        except FileNotFoundError:
            continue
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
