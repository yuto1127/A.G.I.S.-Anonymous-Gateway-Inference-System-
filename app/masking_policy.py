"""Industry packs and content-aware Ollama extraction prompts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.masking_genres import DEFAULT_GENRE, load_genres_list

DEFAULT_INDUSTRY = "general"
_INDUSTRY_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class IndustryPack:
    """Loaded industry masking policy."""

    id: str
    label: str
    focus_genres: tuple[str, ...]
    extends_genres: tuple[tuple[str, str], ...]
    guidance: str


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def get_industries_registry_path() -> Path:
    custom = os.environ.get("AGIS_INDUSTRIES_PATH", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return _repo_data_dir() / "industries.json"


def get_industry_packs_dir() -> Path:
    custom = os.environ.get("AGIS_INDUSTRY_PACKS_DIR", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return _repo_data_dir() / "industry_packs"


def resolve_industry(explicit: str | None = None) -> str:
    """Request arg > AGIS_INDUSTRY env > general."""
    if explicit is not None and explicit.strip():
        raw = explicit.strip().lower()
    else:
        raw = os.environ.get("AGIS_INDUSTRY", DEFAULT_INDUSTRY).strip().lower()
    if not raw:
        raw = DEFAULT_INDUSTRY
    if not _INDUSTRY_ID_RE.match(raw):
        return DEFAULT_INDUSTRY
    return raw


def load_industries_registry() -> list[dict[str, str]]:
    """Load [{id, label, pack}, ...] from industries.json."""
    path = get_industries_registry_path()
    if not path.exists():
        return [{"id": DEFAULT_INDUSTRY, "label": "汎用", "pack": "general.json"}]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("industries.json must be a JSON array")
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"industries.json item {i} must be an object")
        iid = item.get("id")
        label = item.get("label")
        pack = item.get("pack")
        if not isinstance(iid, str) or not isinstance(label, str) or not isinstance(pack, str):
            raise ValueError(f"industries.json item {i} needs id, label, pack strings")
        out.append({"id": iid.strip(), "label": label.strip(), "pack": pack.strip()})
    return out


def list_industries() -> list[tuple[str, str]]:
    """(id, label) for selectboxes."""
    return [(x["id"], x["label"]) for x in load_industries_registry()]


def _pack_path_for_industry(industry_id: str) -> Path | None:
    for entry in load_industries_registry():
        if entry["id"] == industry_id:
            return get_industry_packs_dir() / entry["pack"]
    return None


def load_industry_pack(industry_id: str | None = None) -> IndustryPack:
    """Load pack JSON; fall back to general if unknown."""
    iid = resolve_industry(industry_id)
    path = _pack_path_for_industry(iid)
    if path is None or not path.exists():
        path = get_industry_packs_dir() / "general.json"
        iid = DEFAULT_INDUSTRY
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")

    focus = data.get("focus_genres") or []
    if not isinstance(focus, list):
        raise ValueError("focus_genres must be an array")
    focus_genres = tuple(str(x).strip() for x in focus if str(x).strip())

    extends_raw = data.get("extends_genres") or []
    extends: list[tuple[str, str]] = []
    if isinstance(extends_raw, list):
        for item in extends_raw:
            if isinstance(item, dict):
                code = str(item.get("code", "")).strip()
                label = str(item.get("label", "")).strip()
                if code and label:
                    extends.append((code, label))

    guidance = str(data.get("guidance", "")).strip()
    label = str(data.get("label", iid)).strip() or iid
    pack_id = str(data.get("id", iid)).strip() or iid

    return IndustryPack(
        id=pack_id,
        label=label,
        focus_genres=focus_genres,
        extends_genres=tuple(extends),
        guidance=guidance,
    )


def allowed_genre_codes(pack: IndustryPack) -> list[str]:
    """Base genres from JSON file + industry-specific extensions."""
    base = {g["code"] for g in load_genres_list()}
    codes: list[str] = []
    seen: set[str] = set()
    for code in pack.focus_genres:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    for code, _ in pack.extends_genres:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    for code in sorted(base):
        if code not in seen:
            seen.add(code)
            codes.append(code)
    if DEFAULT_GENRE not in seen:
        codes.append(DEFAULT_GENRE)
    return codes


def _genre_lines_for_prompt(pack: IndustryPack) -> str:
    """Human-readable genre list for the LLM."""
    label_by_code = {g["code"]: g["label"] for g in load_genres_list()}
    for code, lab in pack.extends_genres:
        label_by_code[code] = lab
    lines: list[str] = []
    for code in allowed_genre_codes(pack):
        lab = label_by_code.get(code, code)
        focus = " [重点]" if code in pack.focus_genres else ""
        lines.append(f"- {code}: {lab}{focus}")
    return "\n".join(lines)


def build_extraction_prompt(pack: IndustryPack | None = None, *, masking_mode: str = "placeholder") -> str:
    """System prompt for Ollama: industry-aware extraction (placeholder or abstract mode)."""
    if masking_mode == "abstract":
        return build_abstract_extraction_prompt(pack)
    if pack is None:
        pack = load_industry_pack()
    genres_block = _genre_lines_for_prompt(pack)
    focus_list = ", ".join(pack.focus_genres) if pack.focus_genres else "(なし)"

    return (
        "You extract information that must be kept confidential before sending text to an external LLM. "
        f"Industry context: {pack.label} ({pack.id}). "
        f"{pack.guidance} "
        "Analyze the full text and decide what to mask based on meaning, not only keyword lists. "
        f"Pay special attention to these genre codes: {focus_list}. "
        "Also mask any other sensitive spans implied by the text even if not in the focus list. "
        "Respond with ONLY valid JSON: an array of objects, each with keys "
        '"type" (MUST be one of the allowed genre codes below) '
        'and "value" (exact substring as it appears in the text). '
        "For PERSON: also extract any reading/furigana in parentheses immediately next to "
        "the name (e.g. full-width （カタカナ） or half-width (katakana)), as its own entry "
        "or as one contiguous span with the name—never leave such a reading unlisted. "
        "Use COMPANY_SELF for your organization's name when context implies it; COMPANY_OTHER for third parties. "
        "Use ID_NUMBER for generic IDs; use industry-specific codes (e.g. PATIENT_ID, ACCOUNT_NUMBER) when they fit. "
        "If nothing is found, respond with []. "
        "Do not wrap in markdown fences. Do not add explanations.\n\n"
        "Allowed type codes:\n"
        f"{genres_block}"
    )


def build_abstract_extraction_prompt(pack: IndustryPack | None = None) -> str:
    """Ollama prompt for abstract masking: extract spans + optional role-based replacement."""
    if pack is None:
        pack = load_industry_pack()
    genres_block = _genre_lines_for_prompt(pack)
    focus_list = ", ".join(pack.focus_genres) if pack.focus_genres else "(なし)"

    return (
        "You extract confidential spans from text before it is sent to an external LLM. "
        "Masking uses ABSTRACTION: replacements must keep semantic overview (age bands, regions, generic roles). "
        f"Industry: {pack.label} ({pack.id}). {pack.guidance} "
        f"Focus genres: {focus_list}. "
        "Respond with ONLY valid JSON: an array of objects with keys "
        '"type" (allowed genre code), '
        '"value" (exact substring in the text), '
        '"replacement" (short abstract Japanese phrase to use instead, or null). '
        "Rules for replacement: "
        "PERSON: always set replacement to null (counter labels are assigned by the system). "
        "AGE: always set replacement to null — the system converts ages to bands like 30代前半/30代後半. "
        "Never use 「不明」 or similar for any replacement. "
        "ORG, FACILITY, COMPANY_SELF, COMPANY_OTHER, DIAGNOSIS: provide a short generic role label "
        '(e.g. "大学附属病院", "担当医", "取引先") — never invent real names. '
        "ADDRESS, DATE_OF_BIRTH, ID_NUMBER, EMAIL, PHONE: set replacement to null (system will abstract). "
        "Do not extract standalone label 「年齢」 without a numeric age in the same span. "
        "For PERSON readings in parentheses next to a name, list as separate entries or one span. "
        "If nothing to mask, respond with []. No markdown fences. No explanations.\n\n"
        "Allowed type codes:\n"
        f"{genres_block}"
    )
