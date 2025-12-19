"""
Rules:
- Use async/await
- Do NOT block event loop
- Keep code simple and readable
- Follow existing project structure
"""

"""
Context:
- FastAPI server-rendered application (Jinja2)
- Async SQLAlchemy 2.0
- MySQL database using aiomysql
- Existing tables:
  - video_analysis (id, video_id, video_url, status)
  - highlights (id, analysis_id, start, end, score, category, reason, transcript)
- New table: transcripts (id, video_id, start, duration, text)

Task:
- Implement an async function to save YouTube transcript to database

Requirements:
- Input:
  - video_id: str
  - transcript: list of dicts with keys: start, duration, text
- Use AsyncSession from app.database.db
- Insert transcript rows in BULK (no per-row commit)
- Commit once at the end
- Do NOT block the event loop
- Do NOT use synchronous SQLAlchemy session
- Do NOT print logs
- Raise exceptions on failure

Style:
- Clean, readable, production-ready
- Follow existing project structure
"""
