import asyncio
import json
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from app.services.youtube import get_video_id
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db import get_db
from app.services.transcript import fetch_transcript, save_transcript
from app.services.highlight_ai import detect_highlights
from app.services.user_feedback import submit_clip_feedback
from app.database.models import VideoAnalysis, Highlight

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


async def _save_analysis_and_highlights(
    session: AsyncSession,
    video_id: str,
    video_url: str,
    highlights: list[dict],
) -> list[dict]:
    """Persist VideoAnalysis + Highlight rows and return highlights with DB ids."""
    analysis = VideoAnalysis(video_id=video_id, video_url=video_url, status="done")
    session.add(analysis)
    await session.flush()  # assign analysis.id

    rows: list[Highlight] = []
    for h in highlights:
        row = Highlight(
            analysis_id=analysis.id,
            start=float(h.get("start") or 0.0),
            end=float(h.get("end") or 0.0),
            score=float(h.get("score")) if h.get("score") is not None else None,
            category=str(h.get("category") or "") or None,
            reason=str(h.get("reason") or "") or None,
            transcript=str(h.get("transcript") or "") or None,
        )
        rows.append(row)

    session.add_all(rows)
    await session.flush()  # assign highlight ids
    await session.commit()

    # Attach db ids back to response payload for UI feedback submission
    enriched: list[dict] = []
    for h, row in zip(highlights, rows):
        x = dict(h)
        x["db_id"] = row.id
        x["analysis_id"] = analysis.id
        x["clip_id"] = f"highlight:{row.id}"
        enriched.append(x)
    return enriched

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

            # Persist analysis + highlights
            logs.append("Saving highlights to database...")
            highlights = await _save_analysis_and_highlights(session, video_id, youtube_url, highlights)
            logs.append("Highlights saved to DB.")
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

            logs.append("Saving highlights to database...")
            highlights = await _save_analysis_and_highlights(session, video_id, youtube_url, highlights)
            logs.append("Highlights saved to DB.")
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

            yield _sse("log", "Saving highlights to database...")
            highlights = await _save_analysis_and_highlights(session, video_id, youtube_url, highlights)
            yield _sse("log", "Highlights saved to DB.")

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


@router.post("/feedback_api")
async def feedback_api(
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    """Accept structured user feedback for a clip and persist it.

    Expected JSON:
      {clip_id, rating, weaknesses?, strengths?, notes?}
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        submit_clip_feedback(
            clip_id=str(payload.get("clip_id") or ""),
            rating=payload.get("rating"),
            weaknesses=payload.get("weaknesses"),
            strengths=payload.get("strengths"),
            notes=payload.get("notes"),
            db_session=session,
        )
        await session.commit()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"ok": True})
