"""
graph.py — Supervisor 오케스트레이션

Supervisor 노드가 state.stage 를 보고 다음 전문 에이전트를 결정한다.
각 에이전트는 실행 후 supervisor 로 돌아오며, supervisor 가 흐름을 통제한다.

   START → supervisor ─┬─→ content_analyzer ─┐
                       │←────────────────────┘
                       ├─→ quiz_master ───────┐
                       │←────────────────────┘
                       ├─→ roleplay_partner ──┐  (turn < max_turns 이면 루프)
                       │←────────────────────┘
                       ├─→ feedback_coach ────┐
                       │←────────────────────┘
                       └─→ END

"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .agents import analyzer, quiz_master, roleplay_partner, feedback_coach
from .config import CHECKPOINT_DB_PATH
from .state import LearningState


# ── Supervisor: 다음 에이전트 결정 ───────────────────────────
def supervisor(state: LearningState) -> Dict:
    stage = state.get("stage", "start")
    if stage in ("start", ""):
        route = "content_analyzer"
    elif stage == "analyzed":
        route = "quiz_master"
    elif stage == "graded":
        route = "feedback_coach" if state.get("practice_mode") == "flashcards" else "roleplay_partner"
    elif stage == "roleplay":
        route = "feedback_coach" if state.get("turn_count", 0) >= state.get("max_turns", 3) else "roleplay_partner"
    else:  # done
        route = "END"
    return {"route": route}


def _route(state: LearningState) -> str:
    return state.get("route", "END")


# ── 에이전트 노드 래퍼 ───────────────────────────────────────
def _analyzer_node(state: LearningState) -> Dict:
    return analyzer.run(state)


def _quiz_node(state: LearningState) -> Dict:
    # 그래프 자동 실행: 생성 + (시뮬레이션 답안으로) 채점
    return quiz_master.run(state)


def _roleplay_node(state: LearningState) -> Dict:
    # 자동 실행 시 learner_queue 에서 다음 발화를 소비
    queue: List[str] = state.get("learner_queue", []) or []
    turn = state.get("turn_count", 0)
    utt = queue[turn] if turn < len(queue) else "Thanks, that's all!"
    return roleplay_partner.reply(state, utt)


def _feedback_node(state: LearningState) -> Dict:
    return feedback_coach.run(state)


# ── 그래프 빌드 ─────────────────────────────────────────────
def build_graph(checkpointer=None):
    b = StateGraph(LearningState)
    b.add_node("supervisor", supervisor)
    b.add_node("content_analyzer", _analyzer_node)
    b.add_node("quiz_master", _quiz_node)
    b.add_node("roleplay_partner", _roleplay_node)
    b.add_node("feedback_coach", _feedback_node)

    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", _route, {
        "content_analyzer": "content_analyzer",
        "quiz_master": "quiz_master",
        "roleplay_partner": "roleplay_partner",
        "feedback_coach": "feedback_coach",
        "END": END,
    })
    # 모든 전문 에이전트는 supervisor 로 복귀 → supervisor 가 다음을 결정
    for agent in ("content_analyzer", "quiz_master", "roleplay_partner", "feedback_coach"):
        b.add_edge(agent, "supervisor")

    return b.compile(checkpointer=checkpointer)


def build_persistent_app():
    """SQLite 체크포인트가 붙은 프로덕션용 그래프."""
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)
    try:
        memory.setup()
    except Exception:
        pass
    return build_graph(checkpointer=memory)


# learner_queue 필드는 자동 실행 전용이므로 State 에 동적으로 추가 사용
# (TypedDict total=False 라 런타임 키 주입 허용)
