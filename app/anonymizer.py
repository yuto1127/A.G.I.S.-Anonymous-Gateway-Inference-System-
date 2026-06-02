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


class ExtractedEntity(TypedDict, total=False):
    """Single entity returned by the local LLM (JSON)."""

    type: str
    value: str
    replacement: str | None


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


def _masking_mode() -> str:
    """abstract (default) | placeholder."""
    v = os.environ.get("MASKING_MODE", "abstract").strip().lower()
    if v in ("abstract", "placeholder"):
        return v
    return "abstract"


def _extraction_system_prompt(industry: str | None = None) -> str:
    """Industry-aware extraction instructions (genre codes aligned with masking_genres.json)."""
    from app.masking_policy import build_extraction_prompt, load_industry_pack

    return build_extraction_prompt(
        load_industry_pack(industry), masking_mode=_masking_mode()
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
        repl = item.get("replacement")
        if repl is not None and not isinstance(repl, str):
            repl = None
        elif isinstance(repl, str) and not repl.strip():
            repl = None
        ent: ExtractedEntity = {"type": t_norm, "value": v}
        if repl is not None:
            ent["replacement"] = repl
        out.append(ent)
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


# Parenthetical kana reading immediately after a person name (e.g. 山田太郎（ヤマダ タロウ）).
_KANA_CHUNK = r"[\u3041-\u3096\u30a1-\u30fc・]+"
_PAREN_KANA_READING = re.compile(
    r"^[（(]"
    r"[\s　]*"
    rf"(?:{_KANA_CHUNK}(?:[\s　]+{_KANA_CHUNK})*)"
    r"[\s　]*"
    r"[）)]"
)


def _expand_person_parenthetical_readings(
    raw_text: str, entities: list[ExtractedEntity]
) -> list[ExtractedEntity]:
    """If a PERSON span is followed by （…） reading, add that substring when the model missed it."""
    seen_values: set[str] = {e["value"] for e in entities}
    extra: list[ExtractedEntity] = []
    for e in entities:
        if _sanitize_kind(e["type"]) != "PERSON":
            continue
        name = e["value"]
        if not name.strip():
            continue
        pos = 0
        while True:
            i = raw_text.find(name, pos)
            if i < 0:
                break
            j = i + len(name)
            # re.match(..., pos) still anchors ^ at string start; match on suffix instead.
            m = _PAREN_KANA_READING.match(raw_text[j:])
            if m:
                frag = m.group(0)
                inner = frag[1:-1].strip(" \t　")
                if len(inner) >= 2 and frag not in seen_values:
                    seen_values.add(frag)
                    extra.append({"type": "PERSON", "value": frag})
            pos = i + max(1, len(name))
    return entities + extra


def extract_entities_via_ollama(
    text: str,
    *,
    industry: str | None = None,
    timeout_s: float = 120.0,
) -> list[ExtractedEntity]:
    """Call Ollama chat API and parse extracted confidential spans as structured entities."""
    url = f"{_ollama_base_url()}{OLLAMA_CHAT_PATH}"
    payload = {
        "model": _ollama_model(),
        "stream": False,
        "messages": [
            {"role": "system", "content": _extraction_system_prompt(industry)},
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


def _get_or_create_surface(
    session: Session,
    audit_log_id: int,
    entity_kind: str,
    original_value: str,
    surface: str,
) -> str:
    """Reuse existing Mapping for same audit + kind + value, or persist new surface form."""
    stmt = select(Mapping).where(
        Mapping.audit_log_id == audit_log_id,
        Mapping.entity_kind == entity_kind,
        Mapping.original_value == original_value,
    )
    existing = session.scalars(stmt).first()
    if existing is not None:
        return existing.placeholder

    row = Mapping(
        audit_log_id=audit_log_id,
        entity_kind=entity_kind,
        original_value=original_value,
        placeholder=surface,
    )
    session.add(row)
    session.flush()
    return surface


def _get_or_create_placeholder(
    session: Session,
    audit_log_id: int,
    entity_kind: str,
    original_value: str,
) -> str:
    """Bracket placeholder [KIND_n] for placeholder masking mode."""
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
    return _get_or_create_surface(
        session, audit_log_id, entity_kind, original_value, placeholder
    )


def _masking_source() -> str:
    """auto | rules | both | ollama (ollama is alias for auto)."""
    v = os.environ.get("MASKING_SOURCE", "auto").strip().lower()
    if v == "ollama":
        return "auto"
    if v in ("auto", "rules", "both"):
        return v
    return "auto"


def build_mappings_and_anonymize(
    session: Session,
    audit_log_id: int,
    raw_text: str,
    *,
    industry: str | None = None,
) -> str:
    """Extract spans via rules and/or industry-aware Ollama, persist Mapping rows, return anonymized text."""
    from app.rule_masking import (
        extract_entities_from_rules,
        extract_spans_from_rules,
        merge_rule_and_ollama_entities,
    )

    source = _masking_source()
    entities: list[ExtractedEntity] = []

    # --- Phase: collect entities from DB rules and/or Ollama ---
    if source == "rules":
        entities = extract_entities_from_rules(session, raw_text)
    elif source == "auto":
        entities = extract_entities_via_ollama(raw_text, industry=industry)
    else:
        # both: rule spans win on overlap with Ollama-derived spans
        rule_spans = extract_spans_from_rules(session, raw_text)
        ollama_ents = extract_entities_via_ollama(raw_text, industry=industry)
        entities = merge_rule_and_ollama_entities(rule_spans, ollama_ents, raw_text)

    entities = _expand_person_parenthetical_readings(raw_text, entities)
    entities = _dedupe_entities(entities)

    # --- Phase: replace spans (abstract surface or bracket placeholders) ---
    mode = _masking_mode()
    sorted_entities = sorted(entities, key=lambda e: len(e["value"]), reverse=True)
    text = raw_text
    if mode == "abstract":
        from app.masking_abstract import resolve_abstract_surface, should_skip_masking_entity

        for ent in sorted_entities:
            kind = _sanitize_kind(ent["type"])
            value = ent["value"]
            if not value.strip() or value not in text:
                continue
            if should_skip_masking_entity(ent):
                continue
            surface = resolve_abstract_surface(
                ent,
                session=session,
                audit_log_id=audit_log_id,
                industry=industry,
            )
            _get_or_create_surface(session, audit_log_id, kind, value, surface)
            text = text.replace(value, surface)
    else:
        for ent in sorted_entities:
            kind = _sanitize_kind(ent["type"])
            value = ent["value"]
            if not value.strip() or value not in text:
                continue
            ph = _get_or_create_placeholder(session, audit_log_id, kind, value)
            text = text.replace(value, ph)

    return text


def load_placeholder_map(session: Session, audit_log_id: int) -> dict[str, str]:
    """Build placeholder -> original_value map for de-anonymization."""
    stmt = select(Mapping).where(Mapping.audit_log_id == audit_log_id)
    rows = session.scalars(stmt).all()
    return {m.placeholder: m.original_value for m in rows}


def load_merged_placeholder_map(session: Session, audit_log_ids: list[int]) -> dict[str, str]:
    """Merge placeholder maps across conversation turns (later audit wins on conflict)."""
    merged: dict[str, str] = {}
    for aid in audit_log_ids:
        merged.update(load_placeholder_map(session, aid))
    return merged


def restore_text(anonymized_or_model_text: str, placeholder_to_original: dict[str, str]) -> str:
    """Replace placeholders with originals; longer placeholders first."""
    text = anonymized_or_model_text
    for ph in sorted(placeholder_to_original.keys(), key=len, reverse=True):
        text = text.replace(ph, placeholder_to_original[ph])
    return text
