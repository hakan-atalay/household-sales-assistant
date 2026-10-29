from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    ollama_url: str
    model: str
    num_ctx: int
    feedback_db: Path
    min_group_size: int
    cors_origins: tuple[str, ...]

    @property
    def households_path(self) -> Path:
        return self.data_dir / "households.csv.gz"

    @property
    def purchases_path(self) -> Path:
        return self.data_dir / "purchases.csv.gz"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = os.environ.get
    data_dir = Path(env("DATA_DIR", BACKEND_ROOT / "data"))
    return Settings(
        data_dir=data_dir,
        ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
        model=env("QWEN_MODEL", "qwen3:8b"),
        num_ctx=int(env("NUM_CTX", "8192")),
        feedback_db=Path(env("FEEDBACK_DB", data_dir / "feedback.db")),
        min_group_size=int(env("MIN_GROUP_SIZE", "30")),
        cors_origins=tuple(env("CORS_ORIGINS", "http://localhost:5173").split(",")),
    )
