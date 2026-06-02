"""Resolve human actor label (extensible for future SSO)."""

from __future__ import annotations

import os


def resolve_actor(explicit: str | None = None) -> str:
    """Return non-empty actor string: explicit wins, then AGIS_ACTOR, then anonymous."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("AGIS_ACTOR", "").strip()
    return env if env else "anonymous"
