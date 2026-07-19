# 🎓 LinguaLoop (말문) — AI 언어 학습 파트너

---

## SECTION 1. 워크플로우 소개

**워크플로우 이름**
AI-language-learning-partner (LinguaLoop / 말문)

**한 줄 요약**
사용자가 모국어·학습 언어·유튜브 영상을 선택(입력) → AI가 영상을 분석해 맞춤형 학습 콘텐츠와 역할극을 생성(처리) → 영상 학습과 퀴즈로 표현을 익힌 뒤, AI 파트너와 음성으로 대화하며 표현을 연습하고 피드백·복습을 제공받는(출력) 언어 학습 워크플로우.

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

---

## 🧩 (기술 부록) 아키텍처 개요

Option A: 멀티 에이전트 아키텍처를 채택. Supervisor가 상태를 보고 다음 전문 에이전트를 결정한다.

```
        START → supervisor ─┬─→ ContentAnalyzerAgent   (표현 추출·도구 보강·ChromaDB 신규성 판정)
                            ├─→ QuizMasterAgent        (퀴즈 생성·채점)
                            ├─→ RoleplayPartnerAgent   (페르소나 음성 대화, 턴 루프)
                            ├─→ FeedbackCoachAgent     (피드백·복습·플래시카드)
                            └─→ END
```

## 📡 API 요약

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 현재 모드(mock/실제), 페르소나·TTS·STT 가용 여부 |
| POST | `/youtube/search` | 유튜브 검색 |
| POST | `/session/analyze` | 영상 분석 → 표현 + 퀴즈 생성/저장 |
| POST | `/quiz/grade` | 퀴즈 채점 |
| POST | `/roleplay/start` | AI 파트너가 먼저 여는 오프닝 인사 (+음성) |
| POST | `/roleplay/turn` | 역할극 한 턴 (+음성) |
| POST | `/tts` / `/stt` | 텍스트→음성 / 음성→텍스트 |
| POST | `/session/feedback` | 피드백(사용 표현·교정) + 복습 스케줄링 |
| GET | `/user/{id}/history` · `/reviews` · `/stats` | 학습 이력·복습 예정·통계 |
| GET | `/user/{id}/reviews/all` | 복습 전체 이력 (완료 포함, 사이드바용) |
| POST | `/reviews/{id}/done` | 복습 완료 처리 |
| GET | `/user/{id}/flashcards` | 학습한 표현을 영상별로 묶은 플래시카드 (사이드바용) |

## 🚀 실행 방법

> Whisper 폴백(yt-dlp)을 쓰려면 시스템에 **ffmpeg**도 필요합니다.
> macOS: `brew install ffmpeg` · Ubuntu: `apt install ffmpeg`

```bash
pip install -r requirements.txt

# (선택) 키 설정 — 없으면 mock/sample 로 동작
cp .env.example .env
export $(grep -v '^#' .env | xargs)

uvicorn backend.main:app --reload --port 8000   # 백엔드
streamlit run frontend/streamlit_app.py         # UI (다른 터미널)
python run_demo.py                              # 백엔드 없이 전체 흐름만
pytest -q                                       # 테스트 (15건)
```

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
│   ├── auth.py            # 이메일+비밀번호 인증 (해싱, 이메일 인증코드)
│   └── main.py            # FastAPI
├── frontend/
│   └── streamlit_app.py   # Streamlit UI
├── tests/
│   └── test_nodes.py      # PyTest (46)
├── run_demo.py            # CLI 전체 실행 데모
├── requirements.txt
└── .env.example
```
