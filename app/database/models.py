from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.database.db import Base

class VideoAnalysis(Base):
    __tablename__ = "video_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(64), index=True, nullable=False)
    video_url = Column(String(512), nullable=False)
    status = Column(String(32), default="done", nullable=False)

class Highlight(Base):
    __tablename__ = "highlights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("video_analysis.id"), index=True, nullable=False)

    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    score = Column(Float)

    category = Column(String(64))
    reason = Column(Text)
    transcript = Column(Text)


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(64), index=True, nullable=False)

    start = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    text = Column(Text, nullable=False)


class ClipFeedback(Base):
    """Structured user feedback for evaluated clips.

    Stored as JSON strings in Text columns to avoid DB-specific JSON features.
    """

    __tablename__ = "clip_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identify the clip being evaluated (does not require highlights table usage)
    video_id = Column(String(64), index=True, nullable=False)
    clip_start = Column(Float, nullable=False)
    clip_end = Column(Float, nullable=False)

    # Metadata captured at evaluation time
    category = Column(String(64))
    is_punchline = Column(Integer, default=0, nullable=False)  # 0/1
    hook_position = Column(Float)  # 0.0=start, 1.0=end (optional)

    # Feedback payload
    rating = Column(Integer, nullable=False)  # 1..10
    weaknesses = Column(Text)  # JSON array of WeaknessType
    strengths = Column(Text)   # JSON array of StrengthType
    optional_notes = Column(Text)  # stored only, ignored by logic


class ClipUserFeedback(Base):
    """Structured user responses for each generated clip.

    This is the minimal, future-proof input layer:
    - validates rating + enums in service code
    - stores weaknesses/strengths as JSON arrays in Text columns
    - notes are stored only (not interpreted)
    """

    __tablename__ = "clip_user_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identifier provided by the app (or transcript hash)
    clip_id = Column(String(128), index=True, nullable=False)

    # Feedback payload
    rating = Column(Integer, nullable=False)  # 1..10
    weaknesses = Column(Text)  # JSON array of WeaknessType
    strengths = Column(Text)   # JSON array of StrengthType
    notes = Column(Text)       # stored only, ignored by logic

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

