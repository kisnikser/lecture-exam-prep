"""Runtime configuration, read from the ``.env`` file at the repository root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
QUESTION_SETS_DIR = DATA_DIR / "question-sets"
COURSES_DIR = DATA_DIR / "courses"
PROMPTS_DIR = REPO_ROOT / "pipeline" / "prompts"
LOGS_DIR = REPO_ROOT / "pipeline" / ".logs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gpu_host: str = ""

    llm_provider: str = "vllm"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3.8-Flash-Next-FP8"
    llm_api_key: str = "dummy"
    llm_concurrency: int = 64

    embed_model: str = "BAAI/bge-m3"
    retrieve_top_k: int = 40
    chunk_seconds: float = 90.0
    chunk_overlap_seconds: float = 15.0

    whisper_model: str = "large-v3"
    transcribe_gpus: str = "0"

    vllm_port: int = 8000
    vllm_tp_size: int = 8
    vllm_gpu_mem_util: float = 0.90

    site_base: str = "/"

    @property
    def gpu_list(self) -> list[int]:
        return [int(g) for g in self.transcribe_gpus.split(",") if g.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def course_dir(slug: str) -> Path:
    return COURSES_DIR / slug
