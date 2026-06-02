"""SQLAlchemy ORM models for audit trail and placeholder mappings."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class AuditLog(Base):
    """One end-to-end gateway request: input, anonymized payload, Gemini output, restored output."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    industry: Mapped[str] = mapped_column(String(64), default="general", index=True)
    input_raw: Mapped[str] = mapped_column(Text, default="")
    text_anonymized: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_gemini_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_restored: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    mappings: Mapped[list["Mapping"]] = relationship(
        back_populates="audit_log",
        cascade="all, delete-orphan",
    )


class Mapping(Base):
    """Maps an original span (e.g. a name) to a stable placeholder within one audit log."""

    __tablename__ = "mappings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    audit_log_id: Mapped[int] = mapped_column(
        ForeignKey("audit_logs.id", ondelete="CASCADE"),
        index=True,
    )
    entity_kind: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    original_value: Mapped[str] = mapped_column(Text)
    placeholder: Mapped[str] = mapped_column(String(128))

    audit_log: Mapped["AuditLog"] = relationship(back_populates="mappings")


class MaskingRule(Base):
    """User-defined regex-based masking rule (priority: higher runs first for overlap resolution)."""

    __tablename__ = "masking_rules"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    pattern: Mapped[str] = mapped_column(Text, default="")
    # 秘匿ジャンル（年齢・自社名など）。プレースホルダ種別の既定にも使う。
    genre: Mapped[str] = mapped_column(String(64), default="OTHER", index=True)
    entity_kind: Mapped[str] = mapped_column(String(64), default="")
    priority: Mapped[int] = mapped_column(default=0)
    enabled: Mapped[bool] = mapped_column(default=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ActivityLog(Base):
    """Operational log: who did what (distinct from per-request AuditLog payloads)."""

    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    actor: Mapped[str] = mapped_column(String(256), default="anonymous", index=True)
    action: Mapped[str] = mapped_column(String(64), default="", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    detail_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    audit_log_id: Mapped[int | None] = mapped_column(
        ForeignKey("audit_logs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
