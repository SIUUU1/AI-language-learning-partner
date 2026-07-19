"""
youtube_service.py

학습 콘텐츠를 만들기 위한 자막 확보는 3단계 폴백으로 동작한다.

  1순위) 공식 자막 — youtube-transcript-api
  2순위) Whisper STT — 자막이 없는 영상이면 yt-dlp 로 오디오만 내려받아
         OpenAI Whisper(`whisper-1`)로 직접 받아쓴다.
  3순위) 완전 실패 — 위 두 방법이 모두 실패하면 `TranscriptUnavailableError` 를
         던진다. 호출자(main.py)는 이를 잡아 사용자에게 "다른 영상 URL을
         입력해 달라"는 안내를 보여준다.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List, Optional

from .config import OPENAI_API_KEY, USE_REAL_LLM, USE_REAL_YOUTUBE, YOUTUBE_API_KEY

# ─────────────────────────────────────────────────────────────
# 오프라인 샘플 (키가 없을 때만 사용 — 데모 모드)
# ─────────────────────────────────────────────────────────────
_SAMPLE_SEARCH = [
    {
        "video_id": "cafe_order_demo",
        "title": "Ordering Coffee in English — Real Cafe Conversation",
        "channel": "Everyday English",
        "thumbnail": "",
        "published_at": "2024-01-01T00:00:00Z",
    },
    {
        "video_id": "job_interview_demo",
        "title": "English Job Interview: Questions & Answers",
        "channel": "Business English Pro",
        "thumbnail": "",
        "published_at": "2024-02-01T00:00:00Z",
    },
]
_SAMPLE_IDS = {v["video_id"] for v in _SAMPLE_SEARCH}

SAMPLE_TRANSCRIPT = (
    "Hi there! Welcome. What can I get for you today? "
    "I'll have a medium latte, please. "
    "Sure. Would you like it hot or iced? "
    "Iced, please. Could you make it less sweet? "
    "Of course. Is that for here or to go? "
    "To go, thanks. Anything else? "
    "No, that's all. "
    "Okay, that comes to four fifty. Have a great day!"
)

_LANG_MAP = {"English": "en", "Korean": "ko", "Japanese": "ja", "Spanish": "es",
             "French": "fr", "German": "de", "Chinese": "zh", "한국어": "ko",
             "日本語": "ja", "中文": "zh", "Español": "es", "Français": "fr"}


class TranscriptUnavailableError(Exception):
    """공식 자막·Whisper STT 모두 실패했을 때. 호출자는 사용자에게 다른
    영상 URL을 입력하라고 안내해야 한다."""

    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        self.reason = reason
        super().__init__(f"'{video_id}' 영상의 학습 콘텐츠를 만들 수 없습니다: {reason}")


def _client():
    """YouTube Data API v3 클라이언트."""
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)


def extract_video_id(url_or_id: str) -> str:
    """URL 또는 ID 문자열에서 11자리 video_id 를 추출한다."""
    url_or_id = (url_or_id or "").strip()
    m = re.search(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return url_or_id  # 샘플 id 등은 그대로 통과


# ─────────────────────────────────────────────────────────────
# 1) 검색 — YouTube Data API v3 search.list
# ─────────────────────────────────────────────────────────────
def search_videos(query: str, max_results: int = 6) -> List[Dict]:
    if not USE_REAL_YOUTUBE:
        return [v for v in _SAMPLE_SEARCH]
    try:
        yt = _client()
        resp = (
            yt.search()
            .list(q=query, part="snippet", type="video",
                  maxResults=max_results, relevanceLanguage="en",
                  videoCaption="closedCaption")  # 자막 있는 영상 우선
            .execute()
        )
        out: List[Dict] = []
        for item in resp.get("items", []):
            sn = item["snippet"]
            out.append({
                "video_id": item["id"]["videoId"],
                "title": sn["title"],
                "channel": sn["channelTitle"],
                "thumbnail": sn["thumbnails"]["medium"]["url"],
                "published_at": sn["publishedAt"],
            })
        return out
    except Exception as e:  # pragma: no cover
        print(f"[youtube search fallback] {e}")
        return [v for v in _SAMPLE_SEARCH]


# ─────────────────────────────────────────────────────────────
# 2) 메타데이터 — YouTube Data API v3 videos.list
# ─────────────────────────────────────────────────────────────
def video_metadata(video_id: str) -> Dict:
    if not USE_REAL_YOUTUBE:
        s = next((v for v in _SAMPLE_SEARCH if v["video_id"] == video_id), _SAMPLE_SEARCH[0])
        return {"video_id": video_id, "title": s["title"], "channel": s["channel"], "found": True}
    try:
        yt = _client()
        resp = yt.videos().list(part="snippet,contentDetails", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return {"video_id": video_id, "title": "(unknown)", "channel": "", "found": False}
        sn = items[0]["snippet"]
        return {
            "video_id": video_id,
            "title": sn["title"],
            "channel": sn["channelTitle"],
            "description": sn.get("description", "")[:500],
            "found": True,
        }
    except Exception as e:  # pragma: no cover
        print(f"[youtube metadata fallback] {e}")
        return {"video_id": video_id, "title": "(unknown)", "channel": "", "found": False}


# ─────────────────────────────────────────────────────────────
# 3-a) 1순위: 공식 자막 — youtube-transcript-api
# ─────────────────────────────────────────────────────────────
def _fetch_official_captions(video_id: str, target_language: str) -> Optional[str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        pref = _LANG_MAP.get(target_language, "en")
        api = YouTubeTranscriptApi()
        try:  # 신버전 API
            fetched = api.fetch(video_id, languages=[pref, "en"])
            snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        except AttributeError:  # pragma: no cover  (구버전 API 폴백)
            snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=[pref, "en"])
        text = " ".join(s["text"].replace("\n", " ") for s in snippets)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception as e:  # pragma: no cover
        print(f"[1순위: 공식 자막 실패] {video_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 3-b) 2순위: Whisper STT — 자막이 없는 영상용 최후 수단
# ─────────────────────────────────────────────────────────────
def _transcribe_with_whisper(video_id: str) -> Optional[str]:
    """yt-dlp 로 오디오만 내려받아 OpenAI Whisper 로 직접 받아쓴다.
    OPENAI_API_KEY 가 없거나 yt-dlp/ffmpeg 가 없으면 조용히 None."""
    if not (USE_REAL_LLM and OPENAI_API_KEY):
        return None
    try:
        import yt_dlp
    except ImportError:  # pragma: no cover
        print("[2순위: Whisper 폴백] yt-dlp 미설치 — `pip install yt-dlp` 필요")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        out_tmpl = os.path.join(tmp, f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:  # pragma: no cover
            print(f"[2순위: Whisper 폴백] 오디오 다운로드 실패: {e}")
            return None

        mp3_path = os.path.join(tmp, f"{video_id}.mp3")
        if not os.path.exists(mp3_path):
            candidates = [f for f in os.listdir(tmp) if f.startswith(video_id)]
            if not candidates:
                return None
            mp3_path = os.path.join(tmp, candidates[0])

        # Whisper API 는 파일당 25MB 제한 — 초과하면 포기(다음 폴백으로 넘어감)
        if os.path.getsize(mp3_path) > 25 * 1024 * 1024:
            print("[2순위: Whisper 폴백] 오디오 파일이 25MB 초과 — 처리 불가")
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            with open(mp3_path, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = (resp.text or "").strip()
            return text or None
        except Exception as e:  # pragma: no cover
            print(f"[2순위: Whisper 폴백] 변환 실패: {e}")
            return None


# ─────────────────────────────────────────────────────────────
# 자막 확보 — 3단계 폴백 진입점
# ─────────────────────────────────────────────────────────────
def fetch_transcript(video_id: str, target_language: str = "English") -> Dict:
    """
    반환: {"text": str, "source": "captions" | "whisper" | "sample"}
    실패 시: TranscriptUnavailableError — 호출자가 사용자에게 다른 URL을 안내해야 한다.
    """
    if not USE_REAL_YOUTUBE or video_id in _SAMPLE_IDS:
        return {"text": SAMPLE_TRANSCRIPT, "source": "sample"}

    # 1순위: 공식 자막
    text = _fetch_official_captions(video_id, target_language)
    if text:
        return {"text": text, "source": "captions"}

    # 2순위: Whisper STT (오디오 다운로드 → 받아쓰기)
    text = _transcribe_with_whisper(video_id)
    if text:
        return {"text": text, "source": "whisper"}

    # 3순위: 완전 실패 — 사용자에게 다른 영상을 안내해야 함
    raise TranscriptUnavailableError(
        video_id,
        "자막도 없고, 음성 인식(Whisper)도 실패했어요. 다른 영상을 시도해 주세요.",
    )


def get_video_bundle(url_or_id: str, target_language: str = "English") -> Dict:
    """UI/CLI 편의용: video_id + 메타데이터 + 자막을 한 번에.
    TranscriptUnavailableError 는 호출자에게 그대로 전파된다."""
    vid = extract_video_id(url_or_id)
    meta = video_metadata(vid)
    result = fetch_transcript(vid, target_language)
    meta["transcript"] = result["text"]
    meta["transcript_source"] = result["source"]
    return meta
