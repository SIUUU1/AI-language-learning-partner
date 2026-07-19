"""
tools.py — 에이전트가 호출하는 도구(@tool)
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.tools import tool

USE_WEB_SEARCH = True  # False 로 두면 항상 폴백 예문 사용

_FALLBACK_EXAMPLES = {
    "I'll have ...": "I'll have the soup of the day.",
    "Could you make it ...?": "Could you make it decaf, please?",
    "for here or to go": "Is that for here or to go?",
    "Anything else?": "Anything else for you today?",
    "that comes to ...": "That comes to twelve dollars.",
}

_KB = {
    "I'll have ...": {"definition": "used to place an order", "synonyms": ["I'd like", "Can I get"]},
    "Could you make it ...?": {"definition": "politely request a change", "synonyms": ["Can you make it", "Would you make it"]},
    "for here or to go": {"definition": "dine-in or takeout", "synonyms": ["eat in or take out"]},
    "Anything else?": {"definition": "asking if more is needed", "synonyms": ["Is that all?", "Will that be all?"]},
    "that comes to ...": {"definition": "stating the total price", "synonyms": ["the total is", "that'll be"]},
}


@tool
def dictionary_lookup(expression: str) -> dict:
    """표현의 간단한 정의와 유의어(최대 2개)를 반환한다."""
    return _KB.get(expression, {"definition": "(general expression)", "synonyms": []})


@tool
def example_sentence_search(expression: str) -> list:
    """웹(DuckDuckGo)에서 표현의 실제 사용 예문을 검색한다. 실패 시 폴백."""
    if USE_WEB_SEARCH:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            text = DuckDuckGoSearchRun().invoke(f'"{expression}" example sentence English')
            parts = [p.strip() for p in text.replace("\n", " ").split(". ") if p.strip()]
            if parts:
                return [(p if p.endswith(".") else p + ".") for p in parts[:2]]
        except Exception as e:  # pragma: no cover
            print(f"[web search fallback] {expression}: {e}")
    return [_FALLBACK_EXAMPLES.get(expression, f"Example using '{expression}'.")]


TOOLS = [dictionary_lookup, example_sentence_search]
