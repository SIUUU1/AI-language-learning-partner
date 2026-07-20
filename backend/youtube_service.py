"""
youtube_service.py

학습 콘텐츠를 만들기 위한 자막 확보는 다단계 폴백으로 동작한다.

  0순위) Supadata 외부 Transcript API — SUPADATA_API_KEY 가 있으면 최우선.
         Render/OCI 등 클라우드 IP 차단을 Supadata 인프라가 대신 우회하므로
         배포 환경에서 가장 안정적이다.
  1순위) 공식 자막 — youtube-transcript-api (프록시 설정 시 IP 차단 우회)
  2순위) Whisper STT — 자막이 없는 영상이면 yt-dlp 로 오디오만 내려받아
         OpenAI Whisper(`whisper-1`)로 직접 받아쓴다.
  3순위) 완전 실패 — 위 방법이 모두 실패하면 `TranscriptUnavailableError` 를
         던진다. 호출자(main.py)는 이를 잡아 사용자에게 "다른 영상 URL을
         입력해 달라"는 안내를 보여준다.
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Dict, List, Optional

import requests

from .config import (
    OPENAI_API_KEY, USE_REAL_LLM, USE_REAL_YOUTUBE, YOUTUBE_API_KEY,
    YT_PROXY_PROVIDER, WEBSHARE_PROXY_USERNAME, WEBSHARE_PROXY_PASSWORD,
    YT_HTTP_PROXY, YT_HTTPS_PROXY,
    SUPADATA_API_KEY, SUPADATA_MODE,
)

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


# ─────────────────────────────────────────────────────────────
# 프록시 (클라우드 IP 차단 우회)
# ─────────────────────────────────────────────────────────────
def _resolve_proxy_provider() -> str:
    """명시적으로 지정 안 했으면 설정된 값으로 자동 판별."""
    if YT_PROXY_PROVIDER in ("webshare", "generic"):
        return YT_PROXY_PROVIDER
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        return "webshare"
    if YT_HTTP_PROXY or YT_HTTPS_PROXY:
        return "generic"
    return ""


def _build_proxy_config():
    """youtube-transcript-api(1.x) 용 proxy_config 객체. 미설정이면 None."""
    provider = _resolve_proxy_provider()
    try:
        if provider == "webshare":
            from youtube_transcript_api.proxies import WebshareProxyConfig
            return WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD,
            )
        if provider == "generic":
            from youtube_transcript_api.proxies import GenericProxyConfig
            return GenericProxyConfig(
                http_url=YT_HTTP_PROXY or YT_HTTPS_PROXY,
                https_url=YT_HTTPS_PROXY or YT_HTTP_PROXY,
            )
    except Exception as e:  # pragma: no cover
        print(f"[youtube proxy 설정 실패 → 프록시 없이 진행] {e}")
    return None


def _proxies_dict() -> Optional[Dict[str, str]]:
    """구버전 API / yt-dlp 용 단순 프록시 URL dict. 미설정이면 None.
    (Webshare 회전 프록시는 URL 형태를 직접 주지 않으므로 generic 설정에만 적용된다.)"""
    http_url = YT_HTTP_PROXY or YT_HTTPS_PROXY
    https_url = YT_HTTPS_PROXY or YT_HTTP_PROXY
    if http_url or https_url:
        return {"http": http_url, "https": https_url}
    return None


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
# 3-0) 0순위: Supadata 외부 Transcript API (클라우드 IP 차단 우회)
# ─────────────────────────────────────────────────────────────
_SUPADATA_BASE = "https://api.supadata.ai/v1"


def _supadata_extract_text(data: Optional[dict]) -> Optional[str]:
    """Supadata 응답에서 평문 자막을 뽑는다. content 는 (text=true)면 문자열,
    아니면 [{text, offset, duration}] 배열. 비동기 완료 응답의 중첩도 대비."""
    if not isinstance(data, dict):
        return None
    content = data.get("content")
    if content is None and isinstance(data.get("transcript"), dict):
        content = data["transcript"].get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
    else:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _supadata_poll(job_id: str, headers: Dict[str, str],
                   max_attempts: int = 10, interval: float = 3.0) -> Optional[str]:
    """큰 영상/AI 생성은 202 + jobId 로 비동기 처리 → 완료까지 폴링.
    (Supadata 무료 한도는 10초당 5요청이므로 간격을 둔다.)"""
    url = f"{_SUPADATA_BASE}/transcript/{job_id}"
    for _ in range(max_attempts):
        time.sleep(interval)
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code >= 400:
                print(f"[Supadata poll] {r.status_code}: {r.text[:200]}")
                return None
            data = r.json() or {}
            status = data.get("status")
            if status == "completed":
                return _supadata_extract_text(data)
            if status == "failed":
                print(f"[Supadata poll] 작업 실패: {data.get('error')}")
                return None
            # queued/active → 계속 폴링
        except Exception as e:  # pragma: no cover
            print(f"[Supadata poll 실패] {e}")
            return None
    print("[Supadata poll] 시간 초과")
    return None


def _fetch_via_supadata(video_id: str, target_language: str) -> Optional[str]:
    """Supadata 로 자막을 가져온다. 성공 시 평문 문자열, 실패 시 None."""
    if not SUPADATA_API_KEY:
        return None
    headers = {"x-api-key": SUPADATA_API_KEY}
    params = {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "text": "true",
        "lang": _LANG_MAP.get(target_language, "en"),
        "mode": SUPADATA_MODE or "native",
    }
    try:
        r = requests.get(f"{_SUPADATA_BASE}/transcript",
                         headers=headers, params=params, timeout=30)
        if r.status_code == 202:                       # 비동기 작업
            job_id = (r.json() or {}).get("jobId")
            return _supadata_poll(job_id, headers) if job_id else None
        if r.status_code in (403, 404):                # 비공개/없음/제한
            print(f"[Supadata] 접근 불가({r.status_code}): {video_id}")
            return None
        if r.status_code >= 400:
            print(f"[Supadata] {r.status_code}: {r.text[:200]}")
            return None
        return _supadata_extract_text(r.json())
    except Exception as e:  # pragma: no cover
        print(f"[Supadata 실패] {video_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 3-a) 1순위: 공식 자막 — youtube-transcript-api
# ─────────────────────────────────────────────────────────────
def _fetch_official_captions(video_id: str, target_language: str) -> Optional[str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        pref = _LANG_MAP.get(target_language, "en")
        proxy_config = _build_proxy_config()   # 클라우드 IP 차단 우회 (미설정이면 None)
        api = (YouTubeTranscriptApi(proxy_config=proxy_config)
               if proxy_config is not None else YouTubeTranscriptApi())
        try:  # 신버전 API
            fetched = api.fetch(video_id, languages=[pref, "en"])
            snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        except AttributeError:  # pragma: no cover  (구버전 API 폴백)
            snippets = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[pref, "en"], proxies=_proxies_dict())
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
        _proxies = _proxies_dict()
        if _proxies:   # 클라우드 IP 차단 우회 (generic 프록시에만 적용)
            ydl_opts["proxy"] = _proxies.get("https") or _proxies.get("http")
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
    # 샘플 id 이거나, 실제 자막 소스(YouTube 키 또는 Supadata)가 하나도 없으면 데모 샘플
    has_real_source = USE_REAL_YOUTUBE or bool(SUPADATA_API_KEY)
    if not has_real_source or video_id in _SAMPLE_IDS:
        return {"text": SAMPLE_TRANSCRIPT, "source": "sample"}

    # 0순위: Supadata 외부 API (클라우드 IP 차단 우회 — Render 등 배포 환경 권장)
    text = _fetch_via_supadata(video_id, target_language)
    if text:
        return {"text": text, "source": "supadata"}

    # 1순위: 공식 자막 (youtube-transcript-api; 프록시 설정 시 우회)
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
