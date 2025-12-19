import asyncio
import json
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.services.youtube import get_video_id
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db import get_db
from app.services.transcript import fetch_transcript, save_transcript
from app.services.highlight_ai import detect_highlights

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    youtube_url: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    error = None
    video_id = None
    highlights = []
    logs = []

    try:
        logs.append("Extracting video ID from URL...")
        video_id = get_video_id(youtube_url)
        logs.append(f"Video ID detected: {video_id}")
    except Exception as exc:
        error = str(exc)
        logs.append(f"Failed to extract video ID: {error}")

    if video_id and not error:
        try:
            logs.append("Fetching transcript via pipeline (YouTube API → yt-dlp → Whisper)...")
            # Fetch transcript off-thread to avoid blocking event loop
            transcript = await asyncio.to_thread(fetch_transcript, video_id)
            logs.append(f"Transcript fetched: {len(transcript)} segments")

            # Save transcript rows in bulk (single commit)
            logs.append("Saving transcript to database (bulk insert)...")
            await save_transcript(session, video_id, transcript)
            logs.append("Transcript saved to DB.")

            # Detect highlights heuristically
            logs.append("Detecting highlight candidates (heuristics)...")
            highlights = detect_highlights(transcript)
            logs.append(f"Highlights detected: {len(highlights)} candidates")
        except Exception as exc:
            error = str(exc)
            logs.append(f"Analysis failed: {error}")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "highlights": highlights,
            "video_id": video_id,
            "error": error,
            "logs": logs,
        },
    )


@router.post("/analyze_api")
async def analyze_api(
    youtube_url: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    error = None
    video_id = None
    highlights = []
    logs = []

    try:
        logs.append("Extracting video ID from URL...")
        video_id = get_video_id(youtube_url)
        logs.append(f"Video ID detected: {video_id}")
    except Exception as exc:
        error = str(exc)
        logs.append(f"Failed to extract video ID: {error}")

    if video_id and not error:
        try:
            logs.append("Fetching transcript via pipeline (YouTube API → yt-dlp → Whisper)...")
            transcript = await asyncio.to_thread(fetch_transcript, video_id)
            logs.append(f"Transcript fetched: {len(transcript)} segments")

            logs.append("Saving transcript to database (bulk insert)...")
            await save_transcript(session, video_id, transcript)
            logs.append("Transcript saved to DB.")

            logs.append("Detecting highlight candidates (heuristics)...")
            highlights = detect_highlights(transcript)
            logs.append(f"Highlights detected: {len(highlights)} candidates")
        except Exception as exc:
            error = str(exc)
            logs.append(f"Analysis failed: {error}")

    return JSONResponse({
        "video_id": video_id,
        "highlights": highlights,
        "error": error,
        "logs": logs,
    })


def _sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/analyze_stream")
async def analyze_stream(
    request: Request,
    youtube_url: str,
    session: AsyncSession = Depends(get_db),
):
    async def event_generator():
        logs = []
        try:
            yield _sse("log", "Extracting video ID from URL...")
            video_id = get_video_id(youtube_url)
            yield _sse("log", f"Video ID detected: {video_id}")

            # Fetch transcript (off-thread)
            yield _sse("log", "Fetching transcript via pipeline (YouTube API → yt-dlp → Whisper)...")
            transcript = await asyncio.to_thread(fetch_transcript, video_id)
            yield _sse("log", f"Transcript fetched: {len(transcript)} segments")

            # Save transcript
            yield _sse("log", "Saving transcript to database (bulk insert)...")
            await save_transcript(session, video_id, transcript)
            yield _sse("log", "Transcript saved to DB.")

            # Detect highlights
            yield _sse("log", "Detecting highlight candidates (heuristics)...")
            highlights = detect_highlights(transcript)
            yield _sse("log", f"Highlights detected: {len(highlights)} candidates")

            result = {
                "video_id": video_id,
                "highlights": highlights,
                "error": None,
            }
            yield _sse("result", result)
        except Exception as exc:
            err = str(exc)
            yield _sse("log", f"Analysis failed: {err}")
            yield _sse("result", {"video_id": None, "highlights": [], "error": err})

    from starlette.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")
