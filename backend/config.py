"""
config.py — Configuration + LLM Wrapper

The app is designed to function fully in "mock mode" even without API keys.
 - If `OPENAI_API_KEY` is present, it calls the actual GPT model; otherwise, it returns rule-based mock responses.
 - If `YOUTUBE_API_KEY` is present, it uses the actual YouTube Data API; otherwise, it uses sample data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Path
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

APP_DB_PATH = str(DATA_DIR / "lingualoop.sqlite")            # Application DB
CHECKPOINT_DB_PATH = str(DATA_DIR / "lingualoop_memory.sqlite")  # LangGraph CHECKPOINT
CHROMA_DIR = str(DATA_DIR / "chroma")                       # ChromaDB DIR

# ─────────────────────────────────────────────────────────────
# Key/Mode Flag
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

USE_REAL_LLM = bool(OPENAI_API_KEY)
USE_REAL_YOUTUBE = bool(YOUTUBE_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_llm = None
if USE_REAL_LLM:
    try:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.5)
    except Exception as e:  # pragma: no cover
        print(f"[config] ChatOpenAI initialization failed → switching to mock mode: {e}")
        _llm = None
        USE_REAL_LLM = False


def mode_banner() -> dict:
    """A summary for checking the current execution mode via the UI/log."""
    return {
        "llm": "OpenAI(" + OPENAI_MODEL + ")" if USE_REAL_LLM else "mock",
        "youtube": "YouTube Data API" if USE_REAL_YOUTUBE else "sample",
    }


# ─────────────────────────────────────────────────────────────
# LLM Call Helper (shared by agents)
# ─────────────────────────────────────────────────────────────
def llm_json(system: str, user: str, mock):
    """LLM call returning JSON; returns a mock response upon failure or when offline."""
    if not USE_REAL_LLM:
        return mock
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        r = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        t = r.content.strip().strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        return json.loads(t)
    except Exception as e:  # pragma: no cover
        print(f"[llm_json fallback] {e}")
        return mock


def llm_text(messages, mock: str) -> str:
    """Call an LLM that returns free-form text; return a mock response in case of failure or if the service is offline."""
    if not USE_REAL_LLM:
        return mock
    try:
        return _llm.invoke(messages).content
    except Exception as e:  # pragma: no cover
        print(f"[llm_text fallback] {e}")
        return mock
