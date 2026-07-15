"""
youtube_service.py — Actual data collection layer

Division of roles:
  1) Search / Metadata → YouTube Data API v3 (Official; uses API key)
  2) Transcript (transcript text) → youtube-transcript-api
     · The `captions.download` endpoint of the YouTube Data API requires "video owner OAuth"
       to download the actual subtitle content. Therefore, `youtube-transcript-api` is used
       to obtain subtitles for training, while the official Data API handles metadata and search.
     · An extension point has been included to use Whisper for STT (Speech-to-Text)
       on videos that lack subtitles entirely.

If the key (YOUTUBE_API_KEY) is missing, the service operates gracefully using sample data.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .config import USE_REAL_YOUTUBE, YOUTUBE_API_KEY

# ─────────────────────────────────────────────────────────────
# Offline sample (use only when no key is available)
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


def _client():
    """YouTube Data API v3 Client."""
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video_id from the URL or ID string."""
    url_or_id = (url_or_id or "").strip()
    m = re.search(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return url_or_id  


# ─────────────────────────────────────────────────────────────
# 1) search_videos — YouTube Data API v3 search.list
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
                  videoCaption="closedCaption")  #Prioritize videos with subtitles.
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
# 2) video_metadata — YouTube Data API v3 videos.list
# ─────────────────────────────────────────────────────────────
def video_metadata(video_id: str) -> Dict:
    if not USE_REAL_YOUTUBE:
        s = next((v for v in _SAMPLE_SEARCH if v["video_id"] == video_id), _SAMPLE_SEARCH[0])
        return {"video_id": video_id, "title": s["title"], "channel": s["channel"]}
    try:
        yt = _client()
        resp = yt.videos().list(part="snippet,contentDetails", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return {"video_id": video_id, "title": "(unknown)", "channel": ""}
        sn = items[0]["snippet"]
        return {
            "video_id": video_id,
            "title": sn["title"],
            "channel": sn["channelTitle"],
            "description": sn.get("description", "")[:500],
        }
    except Exception as e:  # pragma: no cover
        print(f"[youtube metadata fallback] {e}")
        return {"video_id": video_id, "title": "(unknown)", "channel": ""}


# ─────────────────────────────────────────────────────────────
# 3) transcript — youtube-transcript-api
# ─────────────────────────────────────────────────────────────
def fetch_transcript(video_id: str, target_language: str = "English") -> str:
    # The sample ID always returns the sample subtitle.
    if video_id in {v["video_id"] for v in _SAMPLE_SEARCH} or not USE_REAL_YOUTUBE:
        return SAMPLE_TRANSCRIPT
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        lang_map = {"English": "en", "Korean": "ko", "Japanese": "ja",
                    "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh"}
        pref = lang_map.get(target_language, "en")
        api = YouTubeTranscriptApi()
        # New API version: fetch(), old version: get_transcript()
        try:
            fetched = api.fetch(video_id, languages=[pref, "en"])
            snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        except AttributeError:  # pragma: no cover
            snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=[pref, "en"])
        text = " ".join(s["text"].replace("\n", " ") for s in snippets)
        return re.sub(r"\s+", " ", text).strip() or SAMPLE_TRANSCRIPT
    except Exception as e:  # pragma: no cover
        print(f"[transcript fallback → sample] {video_id}: {e}")
        return SAMPLE_TRANSCRIPT


def get_video_bundle(url_or_id: str, target_language: str = "English") -> Dict:
    """For UI convenience: video_id, metadata, and subtitles all at once."""
    vid = extract_video_id(url_or_id)
    meta = video_metadata(vid)
    meta["transcript"] = fetch_transcript(vid, target_language)
    return meta
