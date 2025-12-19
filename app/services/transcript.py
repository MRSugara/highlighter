import os
from typing import Iterable, List, Dict, Optional
from sqlalchemy import insert, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from urllib.request import urlopen
from app.database.models import Transcript


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Pipeline untuk mengambil transcript:
    1) youtube_transcript_api
    2) yt-dlp automatic captions (VTT/SRT)
    3) Download audio dan transcribe dengan Faster-Whisper
    """
    # 1) youtube_transcript_api (prefer Indonesian, fallback English)
    api = YouTubeTranscriptApi
    get_one = getattr(api, "get_transcript", None)
    if callable(get_one):
        try:
            return get_one(video_id, languages=["id", "en"])  # type: ignore
        except Exception:
            # continue to next fallback
            pass

    # 2) yt-dlp automatic captions
    auto = _fetch_auto_captions_via_ytdlp(video_id)
    if auto:
        return auto

    # 3) Faster-Whisper transcription from audio
    audio_transcript = _transcribe_audio_with_whisper(video_id)
    if audio_transcript:
        return audio_transcript

    raise RuntimeError("Gagal mengambil transcript dari semua sumber")


def _parse_timestamp(ts: str) -> float:
    # Supports 'HH:MM:SS.mmm' and 'HH:MM:SS,mmm'
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _parse_vtt(vtt_text: str) -> List[Dict]:
    lines = [l.rstrip("\n") for l in vtt_text.splitlines()]
    entries: List[Dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line:
            start_s, end_s = [p.strip() for p in line.split("-->")]
            start = _parse_timestamp(start_s)
            end = _parse_timestamp(end_s)
            text_parts = []
            while i < len(lines) and lines[i].strip():
                text_parts.append(lines[i].strip())
                i += 1
            text = " ".join(text_parts)
            entries.append({"start": start, "duration": max(0.0, end - start), "text": text})
        # skip blank separator
        while i < len(lines) and not lines[i].strip():
            i += 1
    return entries


def _parse_srt(srt_text: str) -> List[Dict]:
    lines = [l.rstrip("\n") for l in srt_text.splitlines()]
    entries: List[Dict] = []
    i = 0
    while i < len(lines):
        # skip index line
        if lines[i].strip().isdigit():
            i += 1
        if i >= len(lines):
            break
        if "-->" in lines[i]:
            start_s, end_s = [p.strip() for p in lines[i].split("-->")]
            start = _parse_timestamp(start_s)
            end = _parse_timestamp(end_s)
            i += 1
            text_parts = []
            while i < len(lines) and lines[i].strip():
                text_parts.append(lines[i].strip())
                i += 1
            text = " ".join(text_parts)
            entries.append({"start": start, "duration": max(0.0, end - start), "text": text})
        # skip blank separator
        while i < len(lines) and not lines[i].strip():
            i += 1
    return entries


def _fetch_auto_captions_via_ytdlp(video_id: str) -> Optional[List[Dict]]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {"quiet": True}
    with YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception:
            return None

    caps = info.get("automatic_captions") or {}
    # prefer Indonesian then English
    for lang in ("id", "id-ID", "en", "en-US"):
        if lang in caps:
            formats = caps[lang]
            # choose vtt first, else srt, else any
            fmt = next((f for f in formats if f.get("ext") == "vtt"), None) or \
                  next((f for f in formats if f.get("ext") == "srt"), None) or \
                  (formats[0] if formats else None)
            if fmt and fmt.get("url"):
                try:
                    with urlopen(fmt["url"]) as resp:
                        data = resp.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if fmt.get("ext") == "srt":
                    entries = _parse_srt(data)
                else:
                    entries = _parse_vtt(data)
                if entries:
                    return entries
    return None


def _transcribe_audio_with_whisper(video_id: str) -> Optional[List[Dict]]:
    # Download best audio only
    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = f"%(id)s.%(ext)s"
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "nocheckcertificate": True,
        # Use env var if ffmpeg is not on PATH
        "ffmpeg_location": os.getenv("FFMPEG_PATH") or None,
    }
    path: Optional[str] = None
    with YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            # Construct produced filename from info
            ext = info.get("ext") or "m4a"
            path = f"{info.get('id')}.{ext}"
        except Exception:
            return None

    if not path:
        return None

    # Transcribe with Faster-Whisper (tiny model for speed)
    try:
        model = WhisperModel("tiny", device="cpu")
        segments, _ = model.transcribe(path, language="id", task="transcribe")
    except Exception:
        return None

    entries: List[Dict] = []
    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        text = str(seg.text).strip()
        entries.append({"start": start, "duration": max(0.0, end - start), "text": text})
    return entries


async def save_transcript(
    session: AsyncSession,
    video_id: str,
    transcript: Iterable[dict]
) -> None:
    # Upsert strategy: delete existing rows for this video_id, then bulk insert
    await session.execute(
        delete(Transcript).where(Transcript.video_id == video_id)
    )

    rows = [
        {
            "video_id": video_id,
            "start": float(t["start"]),
            "duration": float(t["duration"]),
            "text": str(t["text"]),
        }
        for t in transcript
    ]

    if not rows:
        return

    # Do not open a new transaction block; an implicit transaction
    # may already be active on this AsyncSession (due to the SELECT above).
    # Instead, execute and explicitly commit once.
    await session.execute(insert(Transcript), rows)
    await session.commit()
