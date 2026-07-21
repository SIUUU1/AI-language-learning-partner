# 🎓 LinguaLoop (말문) — AI 언어 학습 파트너

> 유튜브 영상 하나로 **표현 학습 → 퀴즈 → 음성 역할극 → 피드백/복습**까지.
> LangGraph 멀티 에이전트가 처음부터 끝까지 자동으로 이어주는 언어 학습 워크플로우.

---

## 📖 소개

**LinguaLoop**은 외국어 영상을 보며 표현을 정리해도 **실제로 써볼 기회가 없어 금방 잊어버리는 문제**를 해결하기 위해 만들어진 AI 언어 학습 파트너입니다.

사용자가 **모국어 · 학습 언어 · 유튜브 영상**만 선택하면, AI가 영상을 분석해 맞춤형 학습 콘텐츠와 역할극을 생성하고, 음성 대화로 표현을 연습시킨 뒤 피드백과 복습까지 이어서 제공합니다.

---

## 📑 배포사이트
https://ai-language-learning-partner-huzvqqk74aw4qqshhxsjsf.streamlit.app/

---

## ✨ 주요 기능

- **🎬 유튜브 기반 표현 학습** — 영상을 검색·선택하면 자막을 분석해 학습할 핵심 표현을 자동으로 추출합니다.
- **📝 자동 퀴즈 생성 및 채점** — 추출한 표현으로 퀴즈를 만들고, 사용자의 답을 AI가 채점합니다.
- **🎭 AI 음성 역할극** — 페르소나별 목소리와 말투를 가진 AI 파트너와 실전 상황을 음성으로 연습합니다.
- **🔊 음성 입출력** — 답변은 `gpt-4o-mini-tts`로 합성하고, 사용자 발화는 `gpt-4o-mini-transcribe`로 받아써 대화 스크립트로 보여줍니다.
- **📈 피드백 · 복습** — 학습 이력을 저장하고, 표현 임베딩 기반으로 플래시카드를 제공합니다.

---

## 🧠 아키텍처 — Supervisor 멀티 에이전트

**LangGraph의 Supervisor 패턴**으로 4개의 전문 에이전트를 오케스트레이션합니다. Supervisor가 현재 상태를 보고 다음에 실행할 에이전트를 결정하며, 진행 상황은 SQLite 체크포인트에 저장되어 중단·재개가 가능합니다.

```
                    ┌──────────────────────┐
   유튜브 영상 선택 ──▶│     Supervisor       │◀── SQLite 체크포인트
                    │  (다음 에이전트 결정)   │
                    └──────────┬───────────┘
             ┌─────────────┬───┴────────┬──────────────┐
             ▼             ▼            ▼              ▼
      ① 표현 분석    ② 퀴즈 생성/채점   ③ 음성 역할극   ④ 피드백/복습
      (자막 분석)     (문제·채점)      (TTS/STT 대화)  (복습 스케줄링)
             │                                          │
             ▼                                          ▼
      ChromaDB (표현 임베딩 메모리)              SQLite (학습 이력·통계)
```

| 에이전트 | 역할 |
| --- | --- |
| ① 표현 분석 | 영상 자막에서 학습할 핵심 표현을 추출 |
| ② 퀴즈 생성/채점 | 표현 기반 퀴즈 생성 및 사용자 답안 채점 |
| ③ 음성 역할극 | 페르소나별 목소리로 실전 대화 연습 (TTS + STT) |
| ④ 피드백/복습 | 학습 피드백 제공 및 복습·플래시카드 스케줄링 |

---

## 🛠️ 기술 스택

| 구분 | 기술 |
| --- | --- |
| 오케스트레이션 | **LangGraph** (Supervisor 기반 멀티 에이전트) |
| LLM | **OpenAI GPT** (표현 추출, 대화 생성) |
| 음성 출력 (TTS) | **gpt-4o-mini-tts** — 페르소나별 목소리/말투 |
| 음성 입력 (STT) | **gpt-4o-mini-transcribe** (whisper-1 폴백) |
| 데이터 수집 | **YouTube Data API v3** (검색/메타데이터), **youtube-transcript-api** (자막) |
| 저장소 | **SQLite** (앱 DB + LangGraph 체크포인트), **ChromaDB** (표현 임베딩) |
| 백엔드 | **FastAPI** |
| 프런트엔드 | **Streamlit** |
| 인증 | **Google OAuth 2.0** |
| 테스트 | **PyTest** (15건) |

---

## 🚀 시작하기

### 1. 요구 사항

- Python 3.11 이상
- (선택) OpenAI API 키, YouTube Data API v3 키 — 없으면 mock/샘플 모드로 동작

### 2. 설치

```bash
git clone https://github.com/SIUUU1/AI-language-learning-partner.git
cd AI-language-learning-partner

pip install -r requirements.txt
```

### 3. 환경 변수 설정 (선택)

```bash
cp .env.example .env
# .env 파일을 열어 필요한 키를 채워 넣으세요. 비워두면 mock/샘플 모드로 동작합니다.
```

### 4. 실행

```bash
# 백엔드 (FastAPI)
uvicorn backend.main:app --reload --port 8000

# UI (다른 터미널에서 Streamlit)
streamlit run frontend/streamlit_app.py

# 백엔드 없이 전체 흐름만 자동 실행
python run_demo.py

# 테스트 (15건)
pytest -q
```

---

## 🔑 환경 변수

`.env.example`을 복사해 `.env`로 저장하세요. **키가 없으면 자동으로 mock/샘플 모드로 끝까지 동작**합니다.

| 변수 | 설명 | 없을 때 동작 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 실제 GPT · TTS · STT 호출용 키 | 규칙 기반 mock 응답 |
| `OPENAI_MODEL` | 사용할 모델 (기본 `gpt-4o-mini`) | — |
| `YOUTUBE_API_KEY` | YouTube Data API v3 키 | 샘플 영상/자막 사용 |
| `LINGUALOOP_API` | Streamlit이 호출할 백엔드 주소 (기본 `http://localhost:8000`) | 기본값 사용 |

> 💡 YouTube API 키는 [Google Cloud Console](https://console.cloud.google.com) → *YouTube Data API v3 사용 설정* → *API 키 발급*에서 받을 수 있습니다.

---

## 🔊🎤 음성 대화 상세

- **출력 (TTS)** — AI 파트너의 답변을 `gpt-4o-mini-tts`로 합성하고 페르소나별 목소리/말투를 적용합니다. 키가 없으면 브라우저 내장 음성으로 폴백합니다.
- **입력 (STT)** — 마이크 녹음을 `gpt-4o-mini-transcribe`로 받아써 채팅 스크립트로 표시합니다 (`whisper-1` 폴백). 키가 없으면 텍스트 입력으로 폴백합니다.

---

## 📡 API 문서

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 현재 모드(mock/실제), 페르소나·TTS·STT 가용 여부 |
| `POST` | `/youtube/search` | 유튜브 검색 |
| `POST` | `/session/analyze` | 영상 분석 → 표현 + 퀴즈 생성/저장 |
| `POST` | `/quiz/grade` | 퀴즈 채점 |
| `POST` | `/roleplay/turn` | 역할극 한 턴 (+음성) |
| `POST` | `/tts` | 텍스트 → 음성 |
| `POST` | `/stt` | 음성 → 텍스트 |
| `POST` | `/session/feedback` | 피드백 + 복습/플래시카드 |
| `GET` | `/user/{id}/history` | 학습 이력 조회 |
| `GET` | `/user/{id}/reviews` | 복습 항목 조회 |
| `GET` | `/user/{id}/stats` | 학습 통계 조회 |

> 백엔드 실행 후 `http://localhost:8000/docs`에서 FastAPI 자동 문서(Swagger UI)로도 확인할 수 있습니다.

---

## 📁 프로젝트 구조

```
AI-language-learning-partner/
├── backend/
│   ├── config.py            # 키/모드, LLM 래퍼(mock 폴백)
│   ├── state.py             # 공유 State + 페르소나
│   ├── youtube_service.py   # YouTube Data API + 자막
│   ├── tools.py             # @tool: 사전/예문 검색
│   ├── agents.py            # ★ 전문 에이전트 4개 (멀티 에이전트)
│   ├── graph.py             # ★ Supervisor 오케스트레이션 + SQLite 체크포인트
│   ├── vectorstore.py       # ChromaDB 표현 메모리
│   ├── tts.py               # gpt-4o-mini-tts 음성 합성 (출력)
│   ├── stt.py               # gpt-4o-mini-transcribe 받아쓰기 (입력)
│   ├── db.py                # 애플리케이션 SQLite
│   └── main.py              # FastAPI
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── tests/
│   └── test_nodes.py        # PyTest (15)
├── run_demo.py              # CLI 전체 실행 데모
├── requirements.txt
└── .env.example
```

---

## 🧪 작성자

**안시우**