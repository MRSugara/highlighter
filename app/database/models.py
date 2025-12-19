from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
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

