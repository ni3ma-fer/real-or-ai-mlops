"""Base PostgreSQL : utilisateurs, clés API (hashées) et journal d'usage."""

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://realorai:realorai@postgres:5432/realorai")

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA-256, jamais la clé en clair
    prefix: Mapped[str] = mapped_column(String(12))  # début de la clé, pour la reconnaître dans une liste
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Usage(Base):
    __tablename__ = "usage"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    endpoint: Mapped[str] = mapped_column(String(50))
    status: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(10), nullable=True)
    prob_ai: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(20), nullable=True)


class Embedding(Base):
    """Empreinte CLIP de chaque image reçue : la matière première de la détection de drift."""
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    model_version: Mapped[str] = mapped_column(String(20))
    prob_ai: Mapped[float] = mapped_column(Float)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # 512 float32 = 2 Ko


class DriftReport(Base):
    """Résultat de chaque passage du job de drift."""
    __tablename__ = "drift_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    window_size: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)  # AUC du classifieur référence vs récent (0,5 = pas de drift)
    threshold: Mapped[float] = mapped_column(Float)
    drift: Mapped[bool] = mapped_column(Boolean)
    share_ai: Mapped[float] = mapped_column(Float)
    reference_version: Mapped[str] = mapped_column(String(20))


def init_db() -> None:
    Base.metadata.create_all(engine)
