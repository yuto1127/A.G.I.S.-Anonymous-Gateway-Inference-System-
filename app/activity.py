"""Append-only operational activity log (separate DB session; survives gateway rollback)."""

from __future__ import annotations

import json
from typing import Any

from app.database import SessionLocal
from app.models import ActivityLog


def write_activity(
    *,
    actor: str,
    action: str,
    summary: str,
    detail: dict[str, Any] | None = None,
    audit_log_id: int | None = None,
) -> None:
    """Persist one activity row (commits immediately)."""
    row = ActivityLog(
        actor=actor[:256],
        action=action[:64],
        summary=summary[:2000] if len(summary) > 2000 else summary,
        detail_json=json.dumps(detail, ensure_ascii=False) if detail else None,
        audit_log_id=audit_log_id,
    )
    session = SessionLocal()
    try:
        session.add(row)
        session.commit()
    finally:
        session.close()
