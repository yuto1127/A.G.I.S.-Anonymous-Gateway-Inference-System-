"""Gateway orchestration: anonymize (Ollama + DB), Gemini call, restore, audit."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.anonymizer import (
    build_mappings_and_anonymize,
    load_placeholder_map,
    restore_text,
)
from app.database import SessionLocal, init_db
from app.models import AuditLog


@dataclass(frozen=True)
class GatewayResult:
    """Outcome of a successful gateway run (persisted row snapshot)."""

    audit_id: int
    created_at: datetime | None
    input_raw: str
    text_anonymized: str
    text_gemini_raw: str
    output_restored: str


def _require_env(name: str) -> str:
    """Return required environment variable or raise with a clear message."""
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def process_gateway(user_text: str) -> GatewayResult:
    """
    End-to-end flow: create audit row, anonymize, call Gemini, restore placeholders, persist.
    On failure, marks AuditLog as failed and re-raises.
    """
    # --- Phase: bootstrap config and schema ---
    load_dotenv()
    init_db()

    session = SessionLocal()
    audit = AuditLog(status="pending", input_raw=user_text)
    session.add(audit)
    session.commit()
    session.refresh(audit)
    audit_id = audit.id

    try:
        # --- Phase: local LLM extraction + DB mappings + replace ---
        try:
            anonymized = build_mappings_and_anonymize(session, audit_id, user_text)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e

        row = session.get(AuditLog, audit_id)
        if row is None:
            raise RuntimeError("AuditLog disappeared after insert")
        row.text_anonymized = anonymized
        row.status = "anonymized"
        session.commit()

        # --- Phase: external model (Gemini) on anonymized text only ---
        _require_env("GEMINI_API_KEY")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # 1.5 は 404 になり得る。2.0-flash は無料枠クォータ枯れで 429 になり得るため、デフォルトは 2.5-flash。
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        gemini_response = model.generate_content(anonymized)
        raw_text = _gemini_response_text(gemini_response)

        row = session.get(AuditLog, audit_id)
        if row is None:
            raise RuntimeError("AuditLog not found after anonymization")
        row.text_gemini_raw = raw_text
        row.status = "sent"
        session.commit()

        # --- Phase: restore placeholders using Mapping table ---
        ph_map = load_placeholder_map(session, audit_id)
        restored = restore_text(raw_text, ph_map)

        row = session.get(AuditLog, audit_id)
        if row is None:
            raise RuntimeError("AuditLog not found before restore persist")
        row.output_restored = restored
        row.status = "restored"
        session.commit()

        final = session.get(AuditLog, audit_id)
        if final is None:
            raise RuntimeError("AuditLog not found after success commit")
        return GatewayResult(
            audit_id=final.id,
            created_at=final.created_at,
            input_raw=final.input_raw,
            text_anonymized=final.text_anonymized or "",
            text_gemini_raw=final.text_gemini_raw or "",
            output_restored=final.output_restored or "",
        )

    except Exception:
        session.rollback()
        row = session.get(AuditLog, audit_id)
        if row is not None:
            row.status = "failed"
            row.error_message = traceback.format_exc()
            session.commit()
        raise
    finally:
        session.close()


def _gemini_response_text(response: object) -> str:
    """Extract text from google-generativeai response object."""
    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = getattr(candidates[0].content, "parts", None) or []
    chunks: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            chunks.append(t)
    out = "".join(chunks).strip()
    if not out:
        raise RuntimeError("Gemini returned empty text")
    return out


def _visible_len(s: str) -> int:
    """Approximate display width (grapheme-unaware; good enough for logs)."""
    return len(s.replace("\n", ""))


def _indent_body(text: str, prefix: str = "  ") -> str:
    """Prefix each line of multiline content for log readability."""
    if not text:
        return prefix + "(空)"
    lines = text.splitlines()
    return "\n".join(prefix + (line if line else " ") for line in lines)


def _rule(width: int = 72) -> str:
    return "─" * width


def _box_line(inner: str, inner_w: int) -> str:
    """One row inside the top summary box (│ … │)."""
    pad = max(0, inner_w - len(inner))
    return "│" + inner + " " * pad + "│"


def _meta_line_boxed(label: str, value: str, *, inner_w: int, label_width: int = 12) -> str:
    inner = f"  {label:<{label_width}} : {value}"
    return _box_line(inner, inner_w)


def format_pipeline_report(
    *,
    audit_id: int,
    status: str,
    created_at: datetime | None,
    input_raw: str,
    text_anonymized: str | None,
    text_gemini_raw: str | None,
    output_restored: str | None,
    include_gemini: bool = True,
) -> str:
    """Human-readable report: original → anonymized → (optional Gemini) → restored."""
    ts = created_at.isoformat() if created_at is not None else "(不明)"
    w = 72

    def section(step: str, title: str, body: str | None, *, empty_label: str) -> list[str]:
        content = body if body else ""
        if not content.strip():
            block = empty_label
            n = 0
        else:
            block = _indent_body(content.rstrip())
            n = _visible_len(content)
        return [
            "",
            f"【{step}】{title}",
            f"    文字数: {n}",
            _rule(w),
            block,
        ]

    inner_w = w - 2
    title_inner = " A.G.I.S. パイプライン結果"
    title_pad = max(0, inner_w - len(title_inner))
    header = [
        "┌" + _rule(inner_w) + "┐",
        "│" + title_inner + " " * title_pad + "│",
        "├" + _rule(inner_w) + "┤",
        _meta_line_boxed("監査ID", str(audit_id), inner_w=inner_w),
        _meta_line_boxed("ステータス", status, inner_w=inner_w),
        _meta_line_boxed("作成時刻", ts, inner_w=inner_w),
        "└" + _rule(inner_w) + "┘",
    ]

    parts: list[str] = header
    parts += section("1", "元データ（入力）", input_raw, empty_label="  (空)")
    parts += section(
        "2",
        "匿名化したデータ（Gemini に送ったテキスト）",
        text_anonymized,
        empty_label="  (未生成 / 失敗前)",
    )
    if include_gemini:
        parts += section(
            "3",
            "Gemini 応答（復元前・プレースホルダのまま）",
            text_gemini_raw,
            empty_label="  (未生成 / 失敗前)",
        )
        step_final = "4"
    else:
        step_final = "3"
    parts += section(
        step_final,
        "復元データ（最終出力）",
        output_restored,
        empty_label="  (未生成 / 失敗前)",
    )
    parts += ["", _rule(w), ""]
    return "\n".join(parts)


def _print_pipeline_report(
    *,
    audit_id: int,
    status: str,
    created_at: datetime | None,
    input_raw: str,
    text_anonymized: str | None,
    text_gemini_raw: str | None,
    output_restored: str | None,
    include_gemini: bool,
    file: object,
) -> None:
    print(
        format_pipeline_report(
            audit_id=audit_id,
            status=status,
            created_at=created_at,
            input_raw=input_raw,
            text_anonymized=text_anonymized,
            text_gemini_raw=text_gemini_raw,
            output_restored=output_restored,
            include_gemini=include_gemini,
        ),
        file=file,
    )


def load_audit(session: Session, audit_id: int) -> AuditLog | None:
    """Fetch a single audit log by primary key."""
    return session.get(AuditLog, audit_id)


def load_latest_audit(session: Session) -> AuditLog | None:
    """Fetch the most recently created audit log."""
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(1)
    return session.scalars(stmt).first()


def cmd_show_audit(audit_id: int | None, *, last: bool, include_gemini: bool) -> int:
    """Print stored pipeline for one audit row."""
    load_dotenv()
    init_db()
    session = SessionLocal()
    try:
        if last:
            row = load_latest_audit(session)
        elif audit_id is not None:
            row = load_audit(session, audit_id)
        else:
            print("audit_id か --last のどちらかを指定してください。", file=sys.stderr)
            return 1
        if row is None:
            print("該当する AuditLog がありません。", file=sys.stderr)
            return 3
        _print_pipeline_report(
            audit_id=row.id,
            status=row.status,
            created_at=row.created_at,
            input_raw=row.input_raw,
            text_anonymized=row.text_anonymized,
            text_gemini_raw=row.text_gemini_raw,
            output_restored=row.output_restored,
            include_gemini=include_gemini,
            file=sys.stdout,
        )
        return 0
    finally:
        session.close()


def _cli(argv: list[str] | None = None) -> int:
    """CLI: run gateway, or show stored audit pipeline."""
    parser = argparse.ArgumentParser(
        description="A.G.I.S. — send text through local anonymization and Gemini, then restore.",
    )
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド（省略時は run）")

    # --- run: 従来どおりメッセージを処理 ---
    run_p = subparsers.add_parser("run", help="ゲートウェイを 1 回実行する")
    run_p.add_argument(
        "message",
        nargs="*",
        help="ユーザー入力（スペース区切りで結合）",
    )
    run_p.add_argument("-m", "--message-one", dest="message_opt", default=None)
    run_p.add_argument(
        "--show-pipeline",
        action="store_true",
        help="元データ・匿名化・Gemini 応答・復元を標準出力に表示する（最終テキストも末尾に含む）",
    )
    run_p.add_argument(
        "--no-gemini-block",
        action="store_true",
        help="--show-pipeline 時に Gemini 応答ブロックを省略する",
    )

    # --- show: DB に保存済みの行を表示 ---
    show_p = subparsers.add_parser("show", help="保存済み AuditLog の内容を表示する")
    show_p.add_argument("--id", type=int, dest="audit_id", default=None, help="audit_logs.id")
    show_p.add_argument("--last", action="store_true", help="最新の 1 件を表示")
    show_p.add_argument(
        "--no-gemini-block",
        action="store_true",
        help="Gemini 応答ブロックを省略する",
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] not in ("run", "show", "-h", "--help"):
        raw_argv = ["run"] + raw_argv
    args = parser.parse_args(raw_argv)

    if args.command == "show":
        if not args.last and args.audit_id is None:
            show_p.print_help()
            return 1
        return cmd_show_audit(
            args.audit_id,
            last=bool(args.last),
            include_gemini=not args.no_gemini_block,
        )

    if args.command != "run":
        parser.print_help()
        return 1

    if args.message_opt:
        text = args.message_opt
    elif args.message:
        text = " ".join(args.message)
    else:
        run_p.print_help()
        return 1

    try:
        result = process_gateway(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.show_pipeline:
        _print_pipeline_report(
            audit_id=result.audit_id,
            status="restored",
            created_at=result.created_at,
            input_raw=result.input_raw,
            text_anonymized=result.text_anonymized,
            text_gemini_raw=result.text_gemini_raw,
            output_restored=result.output_restored,
            include_gemini=not args.no_gemini_block,
            file=sys.stdout,
        )

    print(result.output_restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
