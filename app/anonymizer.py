"""Ollama-driven PII extraction and DB-backed placeholder replacement."""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypedDict

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Mapping


class ExtractedEntity(TypedDict):
    """Single entity returned by the local LLM (JSON)."""

    type: str
    value: str


class AnonymizerParseError(ValueError):
    """Raised when Ollama output cannot be parsed as the expected JSON schema."""


OLLAMA_CHAT_PATH = "/api/chat"


def _running_in_docker() -> bool:
    """True inside a typical Linux container (Docker creates `/.dockerenv`)."""
    return os.path.exists("/.dockerenv")


def _ollama_base_url() -> str:
    """Read Ollama base URL (no trailing slash).

    `host.docker.internal` is for processes *inside* Docker calling the host.
    On the host OS it often does not resolve; map it to loopback unless we are in a container.
    """
    url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if "host.docker.internal" in url and not _running_in_docker():
        url = url.replace("host.docker.internal", "127.0.0.1")
    return url


def _ollama_model() -> str:
    """Read Ollama model name."""
    return os.environ.get("OLLAMA_MODEL", "gemma3:4b")


def _extraction_system_prompt() -> str:
    """Instructions: JSON only, no prose."""
    return (
        "You extract personally identifiable information (PII) from the user's text. "
        "Respond with ONLY valid JSON: an array of objects, each with keys "
        '"type" (uppercase category: PERSON, EMAIL, PHONE, ADDRESS, ORG, ID, OTHER) '
        'and "value" (exact substring as it appears in the text). '
        "If nothing is found, respond with []. "
        "Do not wrap in markdown fences. Do not add explanations."
    )


def _strip_json_fences(raw: str) -> str:
    """Remove optional ```json ... ``` wrapping from model output."""
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_entity_json(raw: str) -> list[ExtractedEntity]:
    """Parse JSON array or {entities: [...]} into a list of ExtractedEntity."""
    cleaned = _strip_json_fences(raw)
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise AnonymizerParseError(f"Invalid JSON from Ollama: {e}") from e

    if isinstance(data, dict) and "entities" in data:
        data = data["entities"]
    if not isinstance(data, list):
        raise AnonymizerParseError("Expected JSON array or object with 'entities' array")

    out: list[ExtractedEntity] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise AnonymizerParseError(f"Item {i} is not an object")
        t = item.get("type")
        v = item.get("value")
        if not isinstance(t, str) or not isinstance(v, str):
            raise AnonymizerParseError(f"Item {i} missing string 'type' or 'value'")
        t_norm = re.sub(r"[^A-Z0-9_]", "_", t.upper()) or "OTHER"
        out.append(ExtractedEntity(type=t_norm, value=v))
    return out


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Drop duplicate (type, value) pairs while preserving order."""
    seen: set[tuple[str, str]] = set()
    result: list[ExtractedEntity] = []
    for e in entities:
        key = (e["type"], e["value"])
        if key in seen:
            continue
        seen.add(key)
        result.append(e)
    return result


def extract_entities_via_ollama(text: str, *, timeout_s: float = 120.0) -> list[ExtractedEntity]:
    """Call Ollama chat API and parse extracted PII as structured entities."""
    url = f"{_ollama_base_url()}{OLLAMA_CHAT_PATH}"
    payload = {
        "model": _ollama_model(),
        "stream": False,
        "messages": [
            {"role": "system", "content": _extraction_system_prompt()},
            {
                "role": "user",
                "content": f"Text:\n{text}\n\nReturn the JSON array only.",
            },
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()

    # /api/chat returns message.content
    message = body.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AnonymizerParseError("Ollama response missing message.content")

    entities = _parse_entity_json(content)
    return _dedupe_entities(entities)


def _sanitize_kind(kind: str) -> str:
    """Use only safe characters inside bracket placeholders."""
    k = re.sub(r"[^A-Z0-9_]", "_", kind.upper())
    return k if k else "OTHER"


def _next_placeholder_index(session: Session, audit_log_id: int, entity_kind: str) -> int:
    """Compute next index for [KIND_n] for this audit log and kind."""
    stmt = select(Mapping.placeholder).where(
        Mapping.audit_log_id == audit_log_id,
        Mapping.entity_kind == entity_kind,
    )
    rows = session.scalars(stmt).all()
    max_idx = 0
    prefix = f"[{entity_kind}_"
    suffix = "]"
    for ph in rows:
        if ph.startswith(prefix) and ph.endswith(suffix):
            inner = ph[len(prefix) : -len(suffix)]
            if inner.isdigit():
                max_idx = max(max_idx, int(inner))
    return max_idx + 1


def _get_or_create_placeholder(
    session: Session,
    audit_log_id: int,
    entity_kind: str,
    original_value: str,
) -> str:
    """Reuse existing Mapping for same audit + kind + value, or allocate a new placeholder."""
    stmt = select(Mapping).where(
        Mapping.audit_log_id == audit_log_id,
        Mapping.entity_kind == entity_kind,
        Mapping.original_value == original_value,
    )
    existing = session.scalars(stmt).first()
    if existing is not None:
        return existing.placeholder

    idx = _next_placeholder_index(session, audit_log_id, entity_kind)
    placeholder = f"[{entity_kind}_{idx}]"
    row = Mapping(
        audit_log_id=audit_log_id,
        entity_kind=entity_kind,
        original_value=original_value,
        placeholder=placeholder,
    )
    session.add(row)
    session.flush()
    return placeholder


def build_mappings_and_anonymize(
    session: Session,
    audit_log_id: int,
    raw_text: str,
) -> str:
    """Extract PII via Ollama, persist Mapping rows, return anonymized text."""
    # --- Phase: call local LLM for structured extraction ---
    entities = extract_entities_via_ollama(raw_text)

    # --- Phase: assign placeholders (reuse or create) in deterministic value length order ---
    # Longer values first to avoid partial replacement collisions.
    sorted_entities = sorted(entities, key=lambda e: len(e["value"]), reverse=True)
    text = raw_text
    for ent in sorted_entities:
        kind = _sanitize_kind(ent["type"])
        value = ent["value"]
        if not value.strip():
            continue
        if value not in text:
            # Model may hallucinate spans not exactly present; skip rather than leaking.
            continue
        ph = _get_or_create_placeholder(session, audit_log_id, kind, value)
        text = text.replace(value, ph)

    return text


def load_placeholder_map(session: Session, audit_log_id: int) -> dict[str, str]:
    """Build placeholder -> original_value map for de-anonymization."""
    stmt = select(Mapping).where(Mapping.audit_log_id == audit_log_id)
    rows = session.scalars(stmt).all()
    return {m.placeholder: m.original_value for m in rows}


def restore_text(anonymized_or_model_text: str, placeholder_to_original: dict[str, str]) -> str:
    """Replace placeholders with originals; longer placeholders first."""
    text = anonymized_or_model_text
    for ph in sorted(placeholder_to_original.keys(), key=len, reverse=True):
        text = text.replace(ph, placeholder_to_original[ph])
    return text
