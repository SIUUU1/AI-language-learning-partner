"""
config.py — 환경설정 + LLM 래퍼
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st  # Streamlit Cloud 환경 대응을 위해 추가

# ─────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

APP_DB_PATH = str(DATA_DIR / "lingualoop.sqlite")            # 애플리케이션 DB
CHECKPOINT_DB_PATH = str(DATA_DIR / "lingualoop_memory.sqlite")  # LangGraph 체크포인트
CHROMA_DIR = str(DATA_DIR / "chroma")                       # ChromaDB 영속 경로

# ─────────────────────────────────────────────────────────────
# 키 / 모드 플래그 (Streamlit Secrets 지원 하이브리드 로직)
# ─────────────────────────────────────────────────────────────
def _get_secret(key_name: str, default: str = "") -> str:
    """Streamlit Secrets에서 먼저 찾고, 없으면 OS 환경변수(os.getenv)에서 가져옵니다."""
    # 1. Streamlit Secrets 확인
    if key_name in st.secrets:
        return str(st.secrets[key_name]).strip()
    # 2. OS 환경 변수 (.env) 확인
    return os.getenv(key_name, default).strip()

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")

# LangChain 등 외부 라이브러리가 내부적으로 os.environ 환경변수를 직접 참조하므로 재등록
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if YOUTUBE_API_KEY:
    os.environ["YOUTUBE_API_KEY"] = YOUTUBE_API_KEY

USE_REAL_LLM = bool(OPENAI_API_KEY)
USE_REAL_YOUTUBE = bool(YOUTUBE_API_KEY)
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")

_llm = None
if USE_REAL_LLM:
    try:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.5)
    except Exception as e:  # pragma: no cover
        print(f"[config] ChatOpenAI 초기화 실패 → mock 모드로 전환: {e}")
        _llm = None
        USE_REAL_LLM = False


def mode_banner() -> dict:
    """현재 실행 모드를 UI/로그에서 확인하기 위한 요약."""
    return {
        "llm": "OpenAI(" + OPENAI_MODEL + ")" if USE_REAL_LLM else "mock",
        "youtube": "YouTube Data API" if USE_REAL_YOUTUBE else "sample",
    }


# ─────────────────────────────────────────────────────────────
# JSON 추출 — GPT가 마크다운 코드펜스나 설명 문장을 덧붙여도 견디도록
# ─────────────────────────────────────────────────────────────
def _extract_json(raw: str):
    """```json 펜스, 앞뒤 설명 문장이 섞여 있어도 최대한 JSON을 뽑아낸다."""
    text = raw.strip()

    # 1) 그대로 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) ```json ... ``` 또는 ``` ... ``` 코드펜스 안쪽만 추출
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) 첫 '[' 또는 '{' 부터 마지막 ']' 또는 '}' 까지 잘라서 시도
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    ends = [i for i in (text.rfind("]"), text.rfind("}")) if i != -1]
    if starts and ends:
        start, end = min(starts), max(ends)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"JSON으로 해석할 수 없는 응답: {text[:200]!r}")


# ─────────────────────────────────────────────────────────────
# LLM 호출 헬퍼 (에이전트들이 공용으로 사용)
# ─────────────────────────────────────────────────────────────
def llm_json(system: str, user: str, mock, context: str = "") -> Tuple[object, Optional[str]]:
    """JSON 을 반환하는 LLM 호출.
    반환: (결과, warning) — 실패/오프라인 시 (mock, 실패 이유) 를 돌려준다."""
    if not USE_REAL_LLM:
        return mock, None
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        r = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return _extract_json(r.content), None
    except Exception as e:
        warning = f"GPT 분석 실패({context or 'unknown'}): {e}. 예시 데이터로 대체했어요."
        print(f"[llm_json fallback] {warning}")
        return mock, warning


def llm_text(messages, mock: str, context: str = "") -> Tuple[str, Optional[str]]:
    """자유 텍스트를 반환하는 LLM 호출.
    반환: (텍스트, warning) — 실패/오프라인 시 (mock, 실패 이유) 를 돌려준다."""
    if not USE_REAL_LLM:
        return mock, None
    try:
        return _llm.invoke(messages).content, None
    except Exception as e:
        warning = f"GPT 응답 실패({context or 'unknown'}): {e}. 예시 응답으로 대체했어요."
        print(f"[llm_text fallback] {warning}")
        return mock, warning
