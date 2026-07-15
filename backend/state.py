"""
state.py — LangGraph State shared by multi-agents

Each specialized agent (node) reads only a portion of this State and updates only the fields it is responsible for.
The Supervisor determines the next agent by examining the `route` field.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, TypedDict

from langgraph.graph.message import add_messages

# Roleplay Partner Persona
PERSONAS: Dict[str, str] = {
    "friend": "a warm, casual friend chatting over coffee",
    "teacher": "a patient language teacher who gently corrects mistakes",
    "interviewer": "a polite job interviewer asking follow-up questions",
    "partner": "a caring significant other having a sweet everyday chat",
    "barista": "a friendly cafe barista taking an order",
}


class LearningState(TypedDict, total=False):
    # ── Input ──
    user_id: str
    session_id: str
    native_language: str
    target_language: str
    video_id: str
    video_title: str
    transcript: str
    persona: str                 # Roleplay Partner Persona Key
    practice_mode: str           # "roleplay" | "flashcards"

    # ── ContentAnalyzerAgent Output ──
    key_expressions: List[Dict]
    enriched_expressions: List[Dict]
    new_expression_count: int    # Number of "new expressions" compared to ChromaDB

    # ── QuizMasterAgent Output ──
    quiz: List[Dict]
    quiz_answers: List[str]
    quiz_score: int

    # ── RoleplayPartnerAgent ──
    messages: Annotated[list, add_messages]
    turn_count: int
    max_turns: int
    learner_utterance: str       # Learner utterances received via the UI (injected at each turn)
    learner_queue: List[str]     # Simulation utterance queue when graph is automatically executed

    # ── FeedbackCoachAgent ──
    feedback: str
    review_list: List[Dict]
    flashcards: List[Dict]

    # ── Shared Memory / Routing ──
    study_history: Annotated[list, operator.add]
    route: str                   # supervisor Routing decision
    stage: str
