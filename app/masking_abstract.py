"""Genre-specific abstraction: preserve meaning while reducing identifiability."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGE_RE = re.compile(r"(\d{1,3})\s*歳")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)")

_AGE_LABEL_ONLY = frozenset({"年齢", "年齢:", "年齢："})


@lru_cache(maxsize=1)
def _load_rules() -> dict[str, Any]:
    custom = os.environ.get("AGIS_ABSTRACTION_RULES_PATH", "").strip()
    path = Path(custom).expanduser().resolve() if custom else _REPO_ROOT / "data" / "abstraction_rules.json"
    if not path.exists():
        return {
            "default_fallback": "秘匿情報",
            "person_prefix_by_industry": {"general": "個人"},
            "genre_strategies": {},
            "role_label_defaults": {},
            "prefecture_to_region": {},
            "numeric_bucket_step": 50,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def person_prefix_for_industry(industry_id: str | None) -> str:
    rules = _load_rules()
    by_ind = rules.get("person_prefix_by_industry") or {}
    iid = (industry_id or "general").strip().lower()
    if isinstance(by_ind, dict) and iid in by_ind:
        return str(by_ind[iid])
    return str(by_ind.get("general", "個人"))


def _strategy_for_kind(kind: str) -> str:
    strategies = _load_rules().get("genre_strategies") or {}
    if isinstance(strategies, dict):
        return str(strategies.get(kind, "fallback"))
    return "fallback"


def parse_age_years(value: str) -> int | None:
    """Extract age in years from spans like `34歳`, `（34歳・女性）`."""
    m = _AGE_RE.search(value)
    if m:
        return int(m.group(1))
    stripped = value.strip()
    if stripped.isdigit() and len(stripped) <= 3:
        return int(stripped)
    if "歳" in value:
        m = _NUMERIC_RE.search(value.replace(",", ""))
        if m:
            return int(float(m.group(1)))
    return None


def contains_age_years(value: str) -> bool:
    return parse_age_years(value) is not None


def should_skip_masking_entity(entity: dict[str, str | None]) -> bool:
    """Skip label-only age tokens (e.g. `年齢` without digits)."""
    from app.anonymizer import _sanitize_kind

    kind = _sanitize_kind(str(entity.get("type", "")))
    value = str(entity.get("value", "")).strip()
    if not value:
        return True
    if kind == "AGE" and not re.search(r"\d", value):
        return True
    if value in _AGE_LABEL_ONLY:
        return True
    return False


def abstract_age(value: str) -> str:
    """Map age to band labels: 20代前半 / 20代後半 (0–4前半, 5–9後半 within decade)."""
    age = parse_age_years(value)
    if age is None:
        return "年齢帯（非特定）"
    if age < 10:
        return "10歳未満"
    if age < 20:
        return "10代"
    if age >= 100:
        return "高齢者"
    decade = (age // 10) * 10
    half = "前半" if age % 10 <= 4 else "後半"
    return f"{decade}代{half}"


def abstract_date_of_birth(value: str) -> str:
    m = _YEAR_RE.search(value)
    if not m:
        return "生年月日（年代不明）"
    year = int(m.group(0))
    decade = (year // 10) * 10
    return f"{decade}年代生"


def abstract_address(value: str) -> str:
    mapping = _load_rules().get("prefecture_to_region") or {}
    if isinstance(mapping, dict):
        for pref, region in mapping.items():
            if pref in value:
                return str(region)
    city_regions = (
        ("札幌", "北海道"),
        ("東京", "関東圏"),
        ("横浜", "関東圏"),
        ("名古屋", "中部"),
        ("大阪", "近畿"),
        ("京都", "近畿"),
        ("福岡", "九州"),
        ("那覇", "沖縄"),
    )
    for token, region in city_regions:
        if token in value:
            return region
    return "国内"


def abstract_numeric_bucket(value: str) -> str:
    step = int(_load_rules().get("numeric_bucket_step") or 50)
    m = _NUMERIC_RE.search(value.replace(",", ""))
    if not m:
        return "数値レンジ（秘匿）"
    num = float(m.group(1))
    low = int(num // step) * step
    high = low + step
    return f"{low}〜{high}台"


def abstract_email(value: str) -> str:
    if "@" in value:
        return "***@example.domain"
    return "メール（秘匿）"


def abstract_phone(value: str) -> str:
    return "電話番号（秘匿）"


def abstract_redact_token(value: str) -> str:
    return "ID（秘匿）"


def abstract_url(value: str) -> str:
    return "URL（秘匿）"


def _role_default(kind: str) -> str:
    defaults = _load_rules().get("role_label_defaults") or {}
    if isinstance(defaults, dict) and kind in defaults:
        return str(defaults[kind])
    return str(_load_rules().get("default_fallback", "秘匿情報"))


def _apply_deterministic_strategy(kind: str, strategy: str, value: str) -> str | None:
    """Return surface text for deterministic strategies, or None if not applicable."""
    if contains_age_years(value) or strategy in ("decade_half", "decade_round") or kind == "AGE":
        return abstract_age(value)
    if strategy == "year_decade" or kind == "DATE_OF_BIRTH":
        return abstract_date_of_birth(value)
    if strategy == "region_block" or kind == "ADDRESS":
        return abstract_address(value)
    if strategy == "redact_email" or kind == "EMAIL":
        return abstract_email(value)
    if strategy == "redact_phone" or kind == "PHONE":
        return abstract_phone(value)
    if strategy in ("redact_token",) or kind in (
        "ID_NUMBER",
        "CREDIT_CARD",
        "PATIENT_ID",
        "INSURANCE_ID",
        "ACCOUNT_NUMBER",
        "CUSTOMER_ID",
        "TRANSACTION_ID",
        "CASE_NUMBER",
        "CONTRACT_ID",
    ):
        return abstract_redact_token(value)
    if strategy == "redact_url" or kind == "URL":
        return abstract_url(value)
    if strategy == "range_bucket":
        return abstract_numeric_bucket(value)
    if strategy == "generic_label":
        return _role_default(kind)
    return None


def _next_person_index(session: Session, audit_log_id: int, prefix: str) -> int:
    stmt = select(Mapping).where(
        Mapping.audit_log_id == audit_log_id,
        Mapping.entity_kind == "PERSON",
    )
    rows = session.scalars(stmt).all()
    max_idx = 0
    for row in rows:
        ph = row.placeholder
        if ph.startswith(prefix) and ph[len(prefix) :].isdigit():
            max_idx = max(max_idx, int(ph[len(prefix) :]))
    return max_idx + 1


def _lookup_existing_surface(
    session: Session,
    audit_log_id: int,
    entity_kind: str,
    original_value: str,
) -> str | None:
    stmt = select(Mapping.placeholder).where(
        Mapping.audit_log_id == audit_log_id,
        Mapping.entity_kind == entity_kind,
        Mapping.original_value == original_value,
    )
    row = session.scalars(stmt).first()
    return row if row else None


def resolve_abstract_surface(
    entity: dict[str, str | None],
    *,
    session: Session,
    audit_log_id: int,
    industry: str | None = None,
) -> str:
    """Compute abstract replacement text for one extracted entity."""
    from app.anonymizer import _sanitize_kind

    kind = _sanitize_kind(str(entity["type"]))
    value = entity["value"]
    existing = _lookup_existing_surface(session, audit_log_id, kind, value)
    if existing is not None:
        return existing

    replacement = entity.get("replacement")
    strategy = _strategy_for_kind(kind)

    # --- Deterministic strategies first (ignore LLM replacement like "不明") ---
    if kind == "PERSON" or strategy == "counter":
        prefix = person_prefix_for_industry(industry)
        idx = _next_person_index(session, audit_log_id, prefix)
        return f"{prefix}{idx}"

    det = _apply_deterministic_strategy(kind, strategy, value)
    if det is not None:
        return det

    # --- Role labels: LLM replacement allowed ---
    if strategy == "role_label":
        if isinstance(replacement, str) and replacement.strip():
            rep = replacement.strip()
            if rep != "不明":
                return rep
        return _role_default(kind)

    # --- Fallback: use replacement only if not the banned placeholder ---
    if isinstance(replacement, str) and replacement.strip():
        rep = replacement.strip()
        if rep != "不明":
            return rep

    return _role_default(kind) if kind in (_load_rules().get("role_label_defaults") or {}) else str(
        _load_rules().get("default_fallback", "秘匿情報")
    )
