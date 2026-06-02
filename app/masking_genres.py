"""Masking genres loaded from JSON (`data/masking_genres.json` by default)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ファイルが無いときに書き出す既定（初回起動用）
_DEFAULT_GENRES: list[dict[str, str]] = [
    {"code": "OTHER", "label": "その他"},
    {"code": "AGE", "label": "年齢"},
    {"code": "WEIGHT", "label": "体重・体積など身体数値"},
    {"code": "COMPANY_SELF", "label": "自社名・自ブランド"},
    {"code": "COMPANY_OTHER", "label": "他社名・取引先名"},
    {"code": "PERSON", "label": "個人名"},
    {"code": "EMAIL", "label": "メールアドレス"},
    {"code": "PHONE", "label": "電話番号"},
    {"code": "ADDRESS", "label": "住所"},
    {"code": "ORG", "label": "組織・部署名（一般）"},
    {"code": "ID_NUMBER", "label": "個人番号・ID・口座など"},
    {"code": "CREDIT_CARD", "label": "カード番号"},
    {"code": "DATE_OF_BIRTH", "label": "生年月日"},
    {"code": "URL", "label": "URL"},
]

DEFAULT_GENRE = "OTHER"
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,62}$")


def get_genres_path() -> Path:
    """JSON path: `MASKING_GENRES_PATH` or `<repo>/data/masking_genres.json`."""
    custom = os.environ.get("MASKING_GENRES_PATH", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "masking_genres.json"


def ensure_genres_file() -> Path:
    """Create parent dirs and default JSON if missing; return path."""
    path = get_genres_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_write_json(path, _DEFAULT_GENRES)
    return path


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_genres_list() -> list[dict[str, str]]:
    """Load `[{code, label}, ...]` from JSON (file order preserved)."""
    path = ensure_genres_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("masking_genres.json must be a JSON array")
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be an object")
        code = item.get("code")
        label = item.get("label")
        if not isinstance(code, str) or not isinstance(label, str):
            raise ValueError(f"Item {i} needs string code and label")
        out.append({"code": code.strip(), "label": label.strip()})
    return out


def save_genres_list(items: list[dict[str, str]]) -> None:
    """Validate, dedupe by code, persist atomically."""
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for it in items:
        code = it["code"].strip()
        label = it["label"].strip()
        validate_genre_entry(code, label)
        if code in seen:
            raise ValueError(f"重複コード: {code}")
        seen.add(code)
        normalized.append({"code": code, "label": label})
    if not normalized:
        raise ValueError("ジャンルを1件以上残してください")
    codes = {x["code"] for x in normalized}
    if DEFAULT_GENRE not in codes:
        raise ValueError(f"必須ジャンル `{DEFAULT_GENRE}` を一覧に含めてください")
    _atomic_write_json(get_genres_path(), normalized)


def validate_genre_entry(code: str, label: str) -> None:
    if not label:
        raise ValueError("ラベルは空にできません")
    if len(label) > 256:
        raise ValueError("ラベルは256文字以内にしてください")
    if not _CODE_RE.match(code):
        raise ValueError(
            "コードは英大文字で始まり、英大文字・数字・アンダースコアのみ（先頭以外は小文字不可）"
        )


def genre_options_tuples() -> list[tuple[str, str]]:
    """`(code, label)` for selectbox."""
    return [(x["code"], x["label"]) for x in load_genres_list()]


def genre_label(code: str) -> str:
    for c, lab in genre_options_tuples():
        if c == code:
            return lab
    return code


def is_known_genre(code: str) -> bool:
    return any(c == code for c, _ in genre_options_tuples())
