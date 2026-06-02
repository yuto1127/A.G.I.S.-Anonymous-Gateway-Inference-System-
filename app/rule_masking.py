"""DB-backed regex masking rules and merge policy with Ollama entities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.anonymizer import ExtractedEntity, _sanitize_kind
from app.models import MaskingRule


class InvalidMaskingPatternError(ValueError):
    """Raised when a regex pattern cannot be compiled."""


def validate_masking_pattern(pattern: str) -> re.Pattern[str]:
    """Compile pattern or raise InvalidMaskingPatternError."""
    try:
        return re.compile(pattern)
    except re.error as e:
        raise InvalidMaskingPatternError(str(e)) from e


def load_enabled_rules(session: Session) -> list[MaskingRule]:
    """Enabled rules: higher priority first, then newer id."""
    stmt = (
        select(MaskingRule)
        .where(MaskingRule.enabled.is_(True))
        .order_by(MaskingRule.priority.desc(), MaskingRule.id.desc())
    )
    return list(session.scalars(stmt).all())


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    value: str
    kind: str
    priority: int
    source: str  # "rule" | "ollama"


def _overlaps(a: _Span, b: _Span) -> bool:
    return a.start < b.end and b.start < a.end


def extract_spans_from_rules(session: Session, text: str) -> list[_Span]:
    """Collect non-overlapping spans from enabled MaskingRule rows (rule wins on ties)."""
    rules = load_enabled_rules(session)
    raw_candidates: list[_Span] = []
    for rule in rules:
        try:
            rx = validate_masking_pattern(rule.pattern)
        except InvalidMaskingPatternError:
            continue
        kind_src = (rule.entity_kind or "").strip() or rule.genre
        kind = _sanitize_kind(kind_src)
        for m in rx.finditer(text):
            raw_candidates.append(
                _Span(
                    start=m.start(),
                    end=m.end(),
                    value=m.group(0),
                    kind=kind,
                    priority=rule.priority,
                    source="rule",
                )
            )

    # Greedy: higher priority, longer span, then leftmost first
    raw_candidates.sort(key=lambda s: (-s.priority, -(s.end - s.start), s.start))
    accepted: list[_Span] = []
    for span in raw_candidates:
        if span.end <= span.start:
            continue
        if any(_overlaps(span, a) for a in accepted):
            continue
        accepted.append(span)
    accepted.sort(key=lambda s: s.start)
    return accepted


def spans_to_entities(spans: list[_Span]) -> list[ExtractedEntity]:
    return [{"type": s.kind, "value": s.value} for s in spans]


def extract_entities_from_rules(session: Session, text: str) -> list[ExtractedEntity]:
    """Public: entities from DB rules only."""
    return spans_to_entities(extract_spans_from_rules(session, text))


def merge_rule_and_ollama_entities(
    rule_spans: list[_Span],
    ollama_entities: list[ExtractedEntity],
    text: str,
) -> list[ExtractedEntity]:
    """Append Ollama entities whose span does not overlap any rule span (first occurrence)."""
    merged_spans = list(rule_spans)
    used_ranges: list[tuple[int, int]] = [(s.start, s.end) for s in merged_spans]

    def overlaps_any(st: int, en: int) -> bool:
        for a, b in used_ranges:
            if st < b and en > a:
                return True
        return False

    for ent in ollama_entities:
        val = ent["value"]
        if not val.strip():
            continue
        pos = 0
        while True:
            i = text.find(val, pos)
            if i < 0:
                break
            st, en = i, i + len(val)
            if not overlaps_any(st, en):
                kind = _sanitize_kind(ent["type"])
                merged_spans.append(_Span(st, en, val, kind, priority=-10**9, source="ollama"))
                used_ranges.append((st, en))
                break
            pos = i + 1

    merged_spans.sort(key=lambda s: s.start)
    out = spans_to_entities(merged_spans)
    ollama_by_value = {(e["type"], e["value"]): e for e in ollama_entities}
    for i, ent in enumerate(out):
        key = (ent["type"], ent["value"])
        if key in ollama_by_value and "replacement" in ollama_by_value[key]:
            ent["replacement"] = ollama_by_value[key].get("replacement")
    return out
