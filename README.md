# 🎓 LinguaLoop (말문) — AI 언어 학습 파트너

---

## SECTION 1. 워크플로우 소개

**워크플로우 이름**
AI-language-learning-partner (LinguaLoop / 말문)

**한 줄 요약**
사용자가 모국어·학습 언어·유튜브 영상을 선택(입력) → AI가 영상을 분석해 맞춤형 학습 콘텐츠와 역할극을 생성(처리) → 영상 학습과 퀴즈로 표현을 익힌 뒤, AI 파트너와 음성으로 대화하며 표현을 연습하고 피드백·복습을 제공받는(출력) 언어 학습 워크플로우.

**해결하는 문제**
외국어 영상을 보며 표현을 정리해도 실제 사용할 기회가 부족해 금방 잊어버리는 문제를 해결한다. 사용자는 모국어와 학습 언어만 선택하면, AI가 맞춤형 학습 콘텐츠 생성부터 실전 대화 연습까지 이어서 제공한다.

---

## SECTION 2. 기술 소개

**사용 기술**

| 구분 | 기술 |
|---|---|
| 오케스트레이션 | LangGraph (Supervisor 기반 멀티 에이전트) |
| LLM | OpenAI GPT (표현 추출, 대화 생성) |
| 음성 출력 | **gpt-4o-mini-tts** (페르소나별 목소리/말투) |
| 음성 입력 | **gpt-4o-mini-transcribe** (whisper-1 폴백) |
| 데이터 수집 | YouTube Data API v3(검색/메타데이터), youtube-transcript-api(자막) |
| 저장소 | SQLite(앱 DB + LangGraph 체크포인트), ChromaDB(표현 임베딩) |
| 백엔드 | FastAPI |
| 프런트엔드 | Streamlit |
| 테스트 | PyTest |

**연동 서비스**
YouTube Data API, OpenAI API(GPT·TTS·STT), SQLite, ChromaDB

**자동화 수준**
반자동 → 완전 자동화된 파이프라인. 사용자가 영상을 선택하면 ① 표현 분석 ② 퀴즈 생성/채점 ③ 음성 역할극 ④ 피드백/복습 스케줄링까지 4개 전문 에이전트가 자동으로 이어서 수행한다. (`run_demo.py` 로 사람 개입 없이 전 과정 자동 실행 가능)

---

## 🔊🎤 음성 대화 (gpt-4o-mini-tts + gpt-4o-mini-transcribe)

- **출력**: AI 파트너 답변을 `gpt-4o-mini-tts`로 합성, 페르소나별 목소리/말투 적용. 키 없으면 브라우저 내장 음성으로 폴백.
- **입력**: 마이크 녹음을 `gpt-4o-mini-transcribe`로 받아써 채팅 스크립트로 표시(whisper-1 폴백). 키 없으면 텍스트 입력으로 폴백.

## 📡 API 요약

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 현재 모드(mock/실제), 페르소나·TTS·STT 가용 여부 |
| POST | `/youtube/search` | 유튜브 검색 |
| POST | `/session/analyze` | 영상 분석 → 표현 + 퀴즈 생성/저장 |
| POST | `/quiz/grade` | 퀴즈 채점 |
| POST | `/roleplay/turn` | 역할극 한 턴 (+음성) |
| POST | `/tts` / `/stt` | 텍스트→음성 / 음성→텍스트 |
| POST | `/session/feedback` | 피드백 + 복습/플래시카드 |
| GET | `/user/{id}/history` · `/reviews` · `/stats` | 학습 이력·복습·통계 |

---

## 🚀 실행 방법

```bash
pip install -r requirements.txt

# (선택) 키 설정 — 없으면 mock/sample 로 동작
cp .env.example .env

uvicorn backend.main:app --reload --port 8000   # 백엔드
streamlit run frontend/streamlit_app.py         # UI (다른 터미널)
python run_demo.py                              # 백엔드 없이 전체 흐름만
pytest -q                                       # 테스트 (15건)
```

---

## 📁 구조

```
lingualoop/
├── backend/
│   ├── config.py          # 키/모드, LLM 래퍼(mock 폴백)
│   ├── state.py           # 공유 State + 페르소나
│   ├── youtube_service.py # YouTube Data API + 자막
│   ├── tools.py           # @tool: 사전/예문 검색
│   ├── agents.py          # ★ 전문 에이전트 4개 (멀티 에이전트)
│   ├── graph.py           # ★ Supervisor 오케스트레이션 + SQLite 체크포인트
│   ├── vectorstore.py     # ChromaDB 표현 메모리
│   ├── tts.py             # gpt-4o-mini-tts 음성 합성 (출력)
│   ├── stt.py             # gpt-4o-mini-transcribe 받아쓰기 (입력)
│   ├── db.py              # 애플리케이션 SQLite
│   └── main.py            # FastAPI
├── frontend/
│   └── streamlit_app.py   # Streamlit UI
├── tests/
│   └── test_nodes.py      # PyTest (15)
├── run_demo.py            # CLI 전체 실행 데모
├── requirements.txt
└── .env.example
```