"""
state.py — LangGraph State
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, TypedDict

from langgraph.graph.message import add_messages

# 역할극 파트너 페르소나 (문서의 "연인/친구/선생님/면접관" 요구 반영)
PERSONAS: Dict[str, str] = {
    "friend": "a warm, casual friend chatting over coffee",
    "teacher": "a patient language teacher who gently corrects mistakes",
    "interviewer": "a polite job interviewer asking follow-up questions",
    "partner": "a caring significant other having a sweet everyday chat",
    "barista": "a friendly cafe barista taking an order",
}


class LearningState(TypedDict, total=False):
    # ── 입력 ──
    user_id: str
    session_id: str
    native_language: str
    target_language: str
    video_id: str
    video_title: str
    transcript: str
    persona: str                 # 역할극 파트너 페르소나 키
    practice_mode: str           # "roleplay" | "flashcards"

    # ── ContentAnalyzerAgent 산출물 ──
    key_expressions: List[Dict]
    enriched_expressions: List[Dict]
    new_expression_count: int    # ChromaDB 대비 "새 표현" 수

    # ── QuizMasterAgent 산출물 ──
    quiz: List[Dict]
    quiz_answers: List[str]
    quiz_score: int

    # ── RoleplayPartnerAgent ──
    messages: Annotated[list, add_messages]
    turn_count: int
    max_turns: int
    learner_utterance: str       # UI 에서 들어온 학습자 발화(턴마다 주입)
    learner_queue: List[str]     # 그래프 자동 실행 시 시뮬레이션 발화 큐

    # ── FeedbackCoachAgent ──
    feedback: str
    review_list: List[Dict]
    flashcards: List[Dict]

    # ── 공용 메모리 / 라우팅 ──
    study_history: Annotated[list, operator.add]
    route: str                   # supervisor 라우팅 결정
    stage: str
    llm_warning: str             # GPT 호출이 실패해 mock으로 대체됐을 때의 사유 (사용자에게 노출)
