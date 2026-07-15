"""
graph.py — Supervisor Orchestration (Assembly component of the multi-agent architecture)

The Supervisor node checks `state.stage` to determine the next specialized agent.
Each agent returns control to the Supervisor after execution, and the Supervisor manages the workflow.

   START → supervisor ─┬─→ content_analyzer ─┐
                       │←────────────────────┘
                       ├─→ quiz_master ───────┐
                       │←────────────────────┘
                       ├─→ roleplay_partner ──┐  (turn < max_turns 이면 루프)
                       │←────────────────────┘
                       ├─→ feedback_coach ────┐
                       │←────────────────────┘
                       └─→ END

Checkpoints are stored in SQLite (SqliteSaver) keyed by `thread_id` → persistent memory.
While the subtitles themselves are provided via the UI/API, this graph is capable of fully automated execution from start to finish.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .agents import analyzer, quiz_master, roleplay_partner, feedback_coach
from .config import CHECKPOINT_DB_PATH
from .state import LearningState


# ── Supervisor: Determine next agent ───────────────────────────
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


# ── agent node wrapper ───────────────────────────────────────
def _analyzer_node(state: LearningState) -> Dict:
    return analyzer.run(state)


def _quiz_node(state: LearningState) -> Dict:
    # Automatic Graph Execution: Generation + Grading (using simulation answers)
    return quiz_master.run(state)


def _roleplay_node(state: LearningState) -> Dict:
    # Consumes the next utterance from the learner_queue during auto-run
    queue: List[str] = state.get("learner_queue", []) or []
    turn = state.get("turn_count", 0)
    utt = queue[turn] if turn < len(queue) else "Thanks, that's all!"
    return roleplay_partner.reply(state, utt)


def _feedback_node(state: LearningState) -> Dict:
    return feedback_coach.run(state)


# ── Graph Build ─────────────────────────────────────────────
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
    # All specialized agents return to the supervisor → the supervisor makes the following decisions.
    for agent in ("content_analyzer", "quiz_master", "roleplay_partner", "feedback_coach"):
        b.add_edge(agent, "supervisor")

    return b.compile(checkpointer=checkpointer)


def build_persistent_app():
    """Production-ready graph with SQLite checkpoints."""
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)
    try:
        memory.setup()
    except Exception:
        pass
    return build_graph(checkpointer=memory)


# Since the `learner_queue` field is reserved for automatic execution, it is used by being dynamically added to the `State`.
# (Allows runtime key injection because TypedDict is total=False)
