"""
agents.py — 멀티 에이전트 아키텍처 

  ┌────────────────────────┬──────────────────────────────────────────────┐
  │ ContentAnalyzerAgent   │ 자막 → 핵심 표현 추출 + 사전/예문 보강 + 신규성 판정 │
  │ QuizMasterAgent        │ 표현 → 빈칸 퀴즈 생성 / 채점                      │
  │ RoleplayPartnerAgent   │ 페르소나(친구·선생님·면접관·연인)로 실전 대화        │
  │ FeedbackCoachAgent     │ 학습자 발화 분석 → 피드백 + 복습/플래시카드 생성     │
  └────────────────────────┴──────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import random
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import llm_json, llm_text
from .state import PERSONAS
from .tools import dictionary_lookup, example_sentence_search
from .vectorstore import get_memory

# ── mock 데이터 (LLM 키 없을 때) ─────────────────────────────
_MOCK_EXPRESSIONS = [
    {"expression": "I'll have ...", "meaning": "~로 할게요 (주문)",
     "explanation": "음식점·카페에서 주문할 때 쓰는 가장 기본적인 표현이에요. 'I want'보다 훨씬 "
                    "공손하게 들려서 실생활에서 정말 많이 씁니다. 뒤에 원하는 메뉴만 바꿔서 붙이면 돼요.",
     "example": "I'll have a medium latte, please."},
    {"expression": "Could you make it ...?", "meaning": "~하게 해주실 수 있나요?",
     "explanation": "이미 주문한 것에 변경 사항을 요청할 때 쓰는 정중한 표현이에요. 'Can you'보다 "
                    "'Could you'가 더 부드럽고 예의 바르게 들려서, 처음 보는 사람에게 쓰기 좋아요.",
     "example": "Could you make it less sweet?"},
    {"expression": "for here or to go", "meaning": "매장에서 드시나요, 포장하시나요?",
     "explanation": "카페·패스트푸드점 직원이 거의 반드시 묻는 질문이에요. 대답할 땐 그냥 'for here' "
                    "또는 'to go'라고 짧게 말해도 자연스러워요.",
     "example": "Is that for here or to go?"},
    {"expression": "Anything else?", "meaning": "더 필요하신 거 있으세요?",
     "explanation": "주문이나 요청을 마무리할 때 점원이 자주 쓰는 표현이에요. 없다고 답할 땐 "
                    "'No, that's all' 또는 'That's it, thanks'라고 하면 됩니다.",
     "example": "No, that's all."},
    {"expression": "that comes to ...", "meaning": "합계가 ~입니다",
     "explanation": "결제 금액을 안내할 때 쓰는 표현이에요. 숫자만 바꿔서 다양한 금액에 활용할 "
                    "수 있고, 카드/현금 결제 여부를 물을 때 바로 이어서 나오는 경우가 많아요.",
     "example": "That comes to four fifty."},
]
_MOCK_QUIZ = [
    {"question": "카페에서 주문할 때: '___ a medium latte, please.'",
     "choices": ["I'll have", "I go", "I making"], "answer": "I'll have"},
    {"question": "덜 달게 요청할 때: 'Could you ___ it less sweet?'",
     "choices": ["make", "made", "to make"], "answer": "make"},
    {"question": "포장 여부를 물을 때: 'Is that for here or ___?'",
     "choices": ["to go", "to went", "go to"], "answer": "to go"},
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
    """자막을 분석해 핵심 표현을 뽑고, 도구로 보강하고, 신규성을 판정한다."""

    name = "content_analyzer"

    def run(self, state: Dict) -> Dict:
        # ① 표현 추출 (모국어로 뜻 + 더 자세한 설명까지 함께 요청)
        native = state.get("native_language", "한국어")
        target = state.get("target_language", "English")
        system = (
            f"You are a {target} language coach. Extract 5-10 useful expressions "
            f"based on the TOPIC and CONTENT of the transcript below. "
            f'Return {{"expression","meaning","explanation","example"}} as JSON '
            f'(a JSON array, nothing else — no markdown fences, no commentary). '
            f'"expression" and "example" MUST be written ENTIRELY in {target} — '
            f"if the transcript is in a different language, translate or adapt "
            f"the idea into a natural {target} expression; never copy foreign-language "
            f"words or phrases into these two fields, and never mix languages within "
            f'a single expression or example. "meaning" is a short translation in '
            f'{native}. "explanation" is 2-3 sentences in {native} explaining when/how '
            "native speakers actually use this expression, nuance, and politeness "
            "level — written for a language learner, not a dictionary."
        )
        user = f"Target language to learn: {target}\nTranscript:\n{state['transcript']}"
        key, warning = llm_json(system, user, mock=_MOCK_EXPRESSIONS, context="표현 추출")

        # ② 도구 보강 + ③ ChromaDB 신규성 판정
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
        history = [f"🔧 analyzed {len(enriched)} expressions ({new_count} new)"]
        if warning:
            history.append(f"⚠️ {warning}")
        return {
            "key_expressions": key,
            "enriched_expressions": enriched,
            "new_expression_count": new_count,
            "study_history": history,
            "llm_warning": warning,
            "stage": "analyzed",
        }


# ════════════════════════════════════════════════════════════
# 2) QuizMasterAgent
# ════════════════════════════════════════════════════════════
class QuizMasterAgent:
    """표현으로 빈칸 퀴즈를 만들고 채점한다."""

    name = "quiz_master"

    def generate(self, state: Dict) -> Dict:
        system = (
            'Make a 3-item multiple-choice quiz from these expressions. '
            'Each item must have exactly 3 choices (1 correct + 2 plausible wrong '
            'options in the same grammatical form). Return a JSON array of '
            '{"question","choices","answer"} — nothing else, no markdown fences, '
            'no commentary. "answer" must be the exact text of the correct choice.'
        )
        src = state.get("enriched_expressions") or state.get("key_expressions") or []
        quiz, warning = llm_json(system, json.dumps(src, ensure_ascii=False),
                                 mock=_MOCK_QUIZ, context="퀴즈 생성")
        # 정답이 항상 첫 번째 보기가 되지 않도록 섞는다
        for q in quiz:
            choices = list(q.get("choices", []))
            random.shuffle(choices)
            q["choices"] = choices
        out = {"quiz": quiz, "stage": "quiz_ready"}
        if warning:
            out["llm_warning"] = warning
            out["study_history"] = [f"⚠️ {warning}"]
        return out

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

    # 그래프 노드용: 생성+채점을 한 번에 (시뮬레이션 답안 사용)
    def run(self, state: Dict) -> Dict:
        out = self.generate(state)
        state = {**state, **out}
        out.update(self.grade(state))
        return out


# ════════════════════════════════════════════════════════════
# 3) RoleplayPartnerAgent
# ════════════════════════════════════════════════════════════
class RoleplayPartnerAgent:
    """선택한 페르소나로 학습자와 실전 대화를 한다 (한 턴 처리)."""

    name = "roleplay_partner"

    def _system(self, state: Dict) -> SystemMessage:
        persona_key = state.get("persona", "barista")
        persona = PERSONAS.get(persona_key, PERSONAS["barista"])
        target = state.get("target_language", "English")
        targets = ", ".join(e["expression"] for e in state.get("key_expressions", []))
        return SystemMessage(content=(
            f"You are {persona}. Roleplay a short, natural conversation "
            f"ENTIRELY in {target}. Respond ONLY in {target} — never mix in any "
            f"other language or code-switch mid-sentence, even if the expressions "
            f"list below or the topic originally came from another language. "
            f"Gently encourage the learner to use these expressions: {targets}. "
            f"Keep replies to 1-2 sentences."))

    def start(self, state: Dict) -> Dict:
        """역할극을 AI가 먼저 시작한다 (오프닝 인사). 학습자 턴은 아직 세지 않는다."""
        convo = [self._system(state), HumanMessage(content=(
            "(Begin the roleplay now as your character. Greet me naturally and start "
            "the scene in 1-2 sentences, inviting me to respond. Do not mention that "
            "this is a roleplay or break character.)"))]
        turn = state.get("turn_count", 0)
        mock = "Hi there! Welcome. What can I get for you today?"
        text, warning = llm_text(convo, mock=mock, context="역할극 시작 인사")
        out = {
            "messages": [AIMessage(content=text)],
            "turn_count": turn,          # 학습자가 아직 말하지 않았으므로 턴은 그대로
            "stage": "roleplay",
        }
        if warning:
            out["llm_warning"] = warning
        return out

    def reply(self, state: Dict, learner_utterance: str) -> Dict:
        """UI 한 턴: 학습자 발화 → 파트너 응답."""
        convo = [self._system(state)] + list(state.get("messages", [])) \
            + [HumanMessage(content=learner_utterance)]
        turn = state.get("turn_count", 0)
        mock = _MOCK_PARTNER_REPLIES[min(turn, len(_MOCK_PARTNER_REPLIES) - 1)]
        text, warning = llm_text(convo, mock=mock, context="역할극 응답")
        out = {
            "messages": [HumanMessage(content=learner_utterance), AIMessage(content=text)],
            "turn_count": turn + 1,
            "stage": "roleplay",
        }
        if warning:
            out["llm_warning"] = warning
        return out

    # 그래프 노드용: state 의 learner_utterance 를 소비
    def run(self, state: Dict) -> Dict:
        utt = state.get("learner_utterance") or "Thanks, bye!"
        return self.reply(state, utt)


# ════════════════════════════════════════════════════════════
# 4) FeedbackCoachAgent
# ════════════════════════════════════════════════════════════
class FeedbackCoachAgent:
    """학습자 발화를 분석해 피드백을 주고, 복습 리스트/플래시카드를 만든다."""

    name = "feedback_coach"

    def feedback(self, state: Dict) -> Dict:
        key_exprs = state.get("key_expressions", [])
        expr_names = [e["expression"] for e in key_exprs]

        convo_lines = []
        for m in state.get("messages", []):
            if isinstance(m, HumanMessage):
                convo_lines.append(f"Learner: {m.content}")
            elif isinstance(m, AIMessage):
                convo_lines.append(f"Partner: {m.content}")
        convo_text = "\n".join(convo_lines)

        native = state.get("native_language", "한국어")
        target = state.get("target_language", "English")

        # mock 폴백: 실제 대화 내용을 볼 수 없으므로, 앞의 절반을 "사용"으로 가정하는
        # 결정적(deterministic) 데모 결과. 진짜 판정은 LLM 모드에서만 가능하다.
        half = max(1, len(expr_names) // 2)
        mock_result = {"used": expr_names[:half], "corrections": [],
                       "overall_comment": "표현을 자연스럽게 잘 사용했어요! 계속 이런 식으로 연습해 보세요."}

        if not key_exprs or not convo_lines:
            result, warning = mock_result, None
        else:
            system = (
                f"You are a {target} conversation coach reviewing a roleplay practice. "
                "Look ONLY at the Learner's lines below (ignore the Partner's lines). "
                'Return JSON: {"used": [...], "corrections": [...], "overall_comment": "..."}. '
                '"used" = the exact strings (verbatim, copied from the target expression list '
                "given below) that the learner actually said or clearly paraphrased — be "
                "generous and count close variations, not just exact matches, but do not "
                'include expressions the learner never attempted. "corrections" = a list of '
                '{"original","issue","suggestion"} for any awkward, unnatural, or grammatically '
                "odd phrases the LEARNER said — even ones unrelated to the target expressions — "
                "so the learner gets real coaching, not just a checklist. Leave corrections empty "
                'if the learner\'s lines were all fine. "overall_comment" is 2-3 encouraging '
                f"sentences written in {native}. Return JSON only, no markdown fences, no commentary."
            )
            user = (
                f"Target expressions:\n{json.dumps(expr_names, ensure_ascii=False)}\n\n"
                f"Conversation:\n{convo_text}"
            )
            result, warning = llm_json(system, user, mock=mock_result, context="피드백 생성")

        used_set = set(result.get("used", []))
        used = [e for e in key_exprs if e["expression"] in used_set]
        missed = [e for e in key_exprs if e["expression"] not in used_set]
        corrections = result.get("corrections", []) or []

        lines = [f"✅ 사용한 표현 {len(used)}개 / ❌ 미사용 {len(missed)}개"]
        if used:
            lines.append("잘 쓴 표현: " + ", ".join(e["expression"] for e in used))
        if missed:
            lines.append("다음엔 이 표현도 써보세요: " + ", ".join(e["expression"] for e in missed))
        if result.get("overall_comment"):
            lines.append("")
            lines.append(result["overall_comment"])

        review = [{"expression": e["expression"], "meaning": e["meaning"],
                   "example": e.get("example", ""), "review_after_days": [1, 3, 7][i % 3]}
                  for i, e in enumerate(missed)]

        history = [f"✅ used {len(used)} / 🔁 review {len(review)}"]
        if warning:
            history.append(f"⚠️ {warning}")
        out = {"feedback": "\n".join(lines), "review_list": review,
              "corrections": corrections, "study_history": history, "stage": "done"}
        if warning:
            out["llm_warning"] = warning
        return out

    def flashcards(self, state: Dict) -> Dict:
        src = state.get("enriched_expressions") or state.get("key_expressions") or []
        cards = [{"front": e["expression"], "back": e["meaning"],
                  "explanation": e.get("explanation", ""),
                  "example": e.get("example", "")} for e in src]
        return {"flashcards": cards,
                "study_history": [f"🃏 {len(cards)} flashcards"],
                "stage": "done"}

    def run(self, state: Dict) -> Dict:
        if state.get("practice_mode") == "flashcards":
            return self.flashcards(state)
        return self.feedback(state)


# 싱글턴 인스턴스 (그래프·API 공용)
analyzer = ContentAnalyzerAgent()
quiz_master = QuizMasterAgent()
roleplay_partner = RoleplayPartnerAgent()
feedback_coach = FeedbackCoachAgent()
