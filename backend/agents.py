"""
agents.py — Multi-agent architecture

Instead of a single, massive prompt or graph, the system is divided into four specialized agents with clearly defined responsibilities.
Each agent manages (1) its own system prompt, (2) its own tools, and (3) updates only its specific state fields.
The Supervisor in `graph.py` orchestrates these agents.

  ┌────────────────────────┬──────────────────────────────────────────────-----------------------------------------┐
  │ ContentAnalyzerAgent   │ Subtitles → Extraction of key expressions                                             │
  │                        │ + Supplementing with dictionary definitions/example sentences + Assessment of novelty │
  │ QuizMasterAgent        │ Expressions → Generate/Grade Fill-in-the-Blank Quiz                                   │
  │ RoleplayPartnerAgent   │ Real-world conversation practice using personas                                       │
  │ FeedbackCoachAgent     │ Analysis of learner utterances → Feedback + Review/Flashcard generation               │
  └────────────────────────┴──────────────────────────────────────────────-----------------------------------------┘

Since each method follows the LangGraph node signature (state -> partial state),
it can be plugged into the graph and also directly reused in FastAPI endpoints.
"""
from __future__ import annotations

import json
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import llm_json, llm_text
from .state import PERSONAS
from .tools import dictionary_lookup, example_sentence_search
from .vectorstore import get_memory

# ── Mock data (when no LLM key is available) ─────────────────────────────
_MOCK_EXPRESSIONS = [
    {"expression": "I'll have ...", "meaning": "~로 할게요 (주문)", "example": "I'll have a medium latte, please."},
    {"expression": "Could you make it ...?", "meaning": "~하게 해주실 수 있나요?", "example": "Could you make it less sweet?"},
    {"expression": "for here or to go", "meaning": "매장/포장 여부", "example": "Is that for here or to go?"},
    {"expression": "Anything else?", "meaning": "더 필요하신 거 있나요?", "example": "No, that's all."},
    {"expression": "that comes to ...", "meaning": "합계가 ~입니다", "example": "That comes to four fifty."},
]
_MOCK_QUIZ = [
    {"question": "주문할 때: '___ a medium latte, please.'", "answer": "I'll have"},
    {"question": "덜 달게: 'Could you ___ it less sweet?'", "answer": "make"},
    {"question": "포장 확인: 'Is that for here or ___?'", "answer": "to go"},
]
_MOCK_PARTNER_REPLIES = [
    "Sure! Would you like it hot or iced?",
    "One iced latte coming up. For here or to go?",
    "Great — that comes to four fifty. Anything else?",
]


# ════════════════════════════════════════════════════════════
# 1) ContentAnalyzerAgent
# ════════════════════════════════════════════════════════════
class ContentAnalyzerAgent:
    """Analyze subtitles to extract key expressions, supplement them using tools, and assess their novelty."""

    name = "content_analyzer"

    def run(self, state: Dict) -> Dict:
        # ① Expression Extraction
        system = ("You are a language coach. Extract 5-10 useful conversational "
                  'expressions from the transcript. Return a JSON list of '
                  '{"expression","meaning","example"}.')
        user = f"Target: {state.get('target_language','English')}\nTranscript:\n{state['transcript']}"
        key = llm_json(system, user, mock=_MOCK_EXPRESSIONS)

        # ② Tool Augmentation + ③ ChromaDB Novelty Assessment
        mem = get_memory()
        uid = state.get("user_id", "anon")
        enriched, new_count = [], 0
        for e in key:
            d = dictionary_lookup.invoke({"expression": e["expression"]})
            ex = example_sentence_search.invoke({"expression": e["expression"]})
            is_new = mem.novelty(uid, e["expression"])
            new_count += int(is_new)
            mem.add(uid, e["expression"], e.get("meaning", ""))
            enriched.append({**e, "definition": d["definition"],
                             "synonyms": d["synonyms"], "extra_examples": ex,
                             "is_new": is_new})
        return {
            "key_expressions": key,
            "enriched_expressions": enriched,
            "new_expression_count": new_count,
            "study_history": [f"🔧 analyzed {len(enriched)} expressions ({new_count} new)"],
            "stage": "analyzed",
        }


# ════════════════════════════════════════════════════════════
# 2) QuizMasterAgent
# ════════════════════════════════════════════════════════════
class QuizMasterAgent:
    """Create a fill-in-the-blank quiz using the expressions and grade it."""

    name = "quiz_master"

    def generate(self, state: Dict) -> Dict:
        system = ('Make a 3-item fill-in-the-blank quiz from these expressions. '
                  'Return a JSON list of {"question","answer"}.')
        src = state.get("enriched_expressions") or state.get("key_expressions") or []
        quiz = llm_json(system, json.dumps(src, ensure_ascii=False), mock=_MOCK_QUIZ)
        return {"quiz": quiz, "stage": "quiz_ready"}

    def grade(self, state: Dict) -> Dict:
        quiz = state.get("quiz", [])
        answers = state.get("quiz_answers", [])
        score = sum(
            1 for q, a in zip(quiz, answers)
            if str(a).strip().lower() == str(q["answer"]).strip().lower()
        )
        return {"quiz_score": score,
                "study_history": [f"📝 quiz {score}/{len(quiz)}"],
                "stage": "graded"}

    # For graph nodes: Generation and scoring in one step (using simulation answers)
    def run(self, state: Dict) -> Dict:
        out = self.generate(state)
        state = {**state, **out}
        out.update(self.grade(state))
        return out


# ════════════════════════════════════════════════════════════
# 3) RoleplayPartnerAgent
# ════════════════════════════════════════════════════════════
class RoleplayPartnerAgent:
    """Engage in a real-life conversation with the learner using the selected persona (process one turn)."""

    name = "roleplay_partner"

    def _system(self, state: Dict) -> SystemMessage:
        persona_key = state.get("persona", "barista")
        persona = PERSONAS.get(persona_key, PERSONAS["barista"])
        targets = ", ".join(e["expression"] for e in state.get("key_expressions", []))
        return SystemMessage(content=(
            f"You are {persona}. Roleplay a short, natural conversation in "
            f"{state.get('target_language','English')} and gently encourage the learner "
            f"to use these expressions: {targets}. Keep replies to 1-2 sentences."))

    def reply(self, state: Dict, learner_utterance: str) -> Dict:
        """UI Turn: Learner utterance → Partner response."""
        convo = [self._system(state)] + list(state.get("messages", [])) \
            + [HumanMessage(content=learner_utterance)]
        turn = state.get("turn_count", 0)
        mock = _MOCK_PARTNER_REPLIES[min(turn, len(_MOCK_PARTNER_REPLIES) - 1)]
        text = llm_text(convo, mock=mock)
        return {
            "messages": [HumanMessage(content=learner_utterance), AIMessage(content=text)],
            "turn_count": turn + 1,
            "stage": "roleplay",
        }

    # For graph nodes: consumes `learner_utterance` from the state.
    def run(self, state: Dict) -> Dict:
        utt = state.get("learner_utterance") or "Thanks, bye!"
        return self.reply(state, utt)


# ════════════════════════════════════════════════════════════
# 4) FeedbackCoachAgent
# ════════════════════════════════════════════════════════════
class FeedbackCoachAgent:
    """Analyze learner utterances to provide feedback, and create review lists or flashcards."""

    name = "feedback_coach"

    def feedback(self, state: Dict) -> Dict:
        learner_text = " ".join(
            m.content for m in state.get("messages", [])
            if isinstance(m, HumanMessage)
        ).lower()
        used, missed = [], []
        for e in state.get("key_expressions", []):
            head = e["expression"].split(" ...")[0].split(" or ")[0].strip().lower()
            (used if head and head in learner_text else missed).append(e)
        lines = [f"✅ 사용한 표현 {len(used)}개 / ❌ 미사용 {len(missed)}개"]
        if used:
            lines.append("잘 쓴 표현: " + ", ".join(e["expression"] for e in used))
        if missed:
            lines.append("다음엔 이 표현도 써보세요: " + ", ".join(e["expression"] for e in missed))
        review = [{"expression": e["expression"], "meaning": e["meaning"],
                   "example": e.get("example", ""), "review_after_days": [1, 3, 7][i % 3]}
                  for i, e in enumerate(missed)]
        return {"feedback": "\n".join(lines), "review_list": review,
                "study_history": [f"✅ used {len(used)} / 🔁 review {len(review)}"],
                "stage": "done"}

    def flashcards(self, state: Dict) -> Dict:
        src = state.get("enriched_expressions") or state.get("key_expressions") or []
        cards = [{"front": e["expression"], "back": e["meaning"],
                  "example": e.get("example", "")} for e in src]
        return {"flashcards": cards,
                "study_history": [f"🃏 {len(cards)} flashcards"],
                "stage": "done"}

    def run(self, state: Dict) -> Dict:
        if state.get("practice_mode") == "flashcards":
            return self.flashcards(state)
        return self.feedback(state)


# Singleton instance (shared by graph and API)
analyzer = ContentAnalyzerAgent()
quiz_master = QuizMasterAgent()
roleplay_partner = RoleplayPartnerAgent()
feedback_coach = FeedbackCoachAgent()
