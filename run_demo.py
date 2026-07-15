"""
run_demo.py — 백엔드/UI 없이 "에이전트 처음부터 끝까지" 자동 실행 데모

Supervisor 그래프가 4개 전문 에이전트를 순서대로 오케스트레이션하고,
SQLite 체크포인트에 상태를 저장한 뒤 다시 읽어와 영속 메모리를 확인한다.

실행:  python run_demo.py
"""
from backend.graph import build_persistent_app
from backend.youtube_service import get_video_bundle
from langchain_core.messages import HumanMessage, AIMessage

app = build_persistent_app()
cfg = {"configurable": {"thread_id": "demo-learner"}}

bundle = get_video_bundle("cafe_order_demo", "English")   # 실제 키 있으면 진짜 영상
init = {
    "user_id": "demo-learner", "session_id": "demo",
    "native_language": "한국어", "target_language": "English",
    "video_id": bundle["video_id"], "video_title": bundle["title"],
    "transcript": bundle["transcript"],
    "persona": "friend", "practice_mode": "roleplay",
    "quiz_answers": ["I'll have", "make", "here"],   # Q3 오답
    "messages": [], "turn_count": 0, "max_turns": 3,
    "learner_queue": ["Hi, I'll have a latte, please.",
                      "Could you make it iced?", "To go, thanks."],
    "study_history": [], "stage": "start",
}

res = app.invoke(init, cfg)

print("=" * 60, "\n① 표현 + 🔧 도구 보강\n", "=" * 60, sep="")
for e in res["enriched_expressions"]:
    tag = "🆕" if e.get("is_new") else "🔁"
    print(f"{tag} {e['expression']:<24} | {e['meaning']}  ({', '.join(e['synonyms'])})")

print("\n" + "=" * 60, f"\n② 퀴즈 {res['quiz_score']}/{len(res['quiz'])}\n", "=" * 60, sep="")
print("\n" + "=" * 60, "\n③ 역할극 (friend)\n", "=" * 60, sep="")
for m in res["messages"]:
    who = "🧑" if isinstance(m, HumanMessage) else "🤖"
    print(f"{who} {m.content}")

print("\n" + "=" * 60, "\n④ 피드백\n", "=" * 60, sep="")
print(res["feedback"])

print("\n" + "=" * 60, "\n⑤ 복습\n", "=" * 60, sep="")
for r in res["review_list"]:
    print(f"- {r['expression']:<24} → {r['review_after_days']}일 후")

# 💾 SQLite 영속 메모리 확인
saved = app.get_state(cfg)
print("\n" + "=" * 60, "\n💾 SQLite 체크포인트에 저장된 학습 이력\n", "=" * 60, sep="")
for h in saved.values["study_history"]:
    print("  •", h)
