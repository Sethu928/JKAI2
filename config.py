import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Secrets & endpoints ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://192.168.1.142:1234/v1")
LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct-1m")
KS_PASSWORD: str = os.getenv("KS_PASSWORD", "changeme")

OPENAI_AVAILABLE: bool = bool(OPENAI_API_KEY)

# --- Paths ---
DB_PATH: str = "memory/jkai2.db"
LOG_DIR: str = "logs"

# --- LLM tuning ---
LLM_MAX_TOKENS: int = 800
LLM_TIMEOUT: int = 30
LLM_TEMPERATURE: float = 0.7

# --- Agent behaviour ---
WORKER_INTERVAL: int = 60
ERROR_MAX_TRIES: int = 3


def ensure_dirs() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


ensure_dirs()
