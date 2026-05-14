"""A.G.I.S. ローカル Web GUI（Streamlit）: チャット・DB 確認・パイプライン一括表示。"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# streamlit run gui/streamlit_app.py をリポジトリルートから実行することを想定
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select

from app.database import SessionLocal, init_db
from app.main import format_pipeline_report, process_gateway
from app.models import AuditLog, Mapping


def _bootstrap() -> None:
    load_dotenv(override=False)
    init_db()


def _snippet(text: str | None, max_len: int = 48) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _list_audits(limit: int = 200) -> list[AuditLog]:
    s = SessionLocal()
    try:
        q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        return list(s.scalars(q).all())
    finally:
        s.close()


def _get_audit(audit_id: int) -> AuditLog | None:
    s = SessionLocal()
    try:
        return s.get(AuditLog, audit_id)
    finally:
        s.close()


def _list_mappings(audit_id: int) -> list[Mapping]:
    s = SessionLocal()
    try:
        q = select(Mapping).where(Mapping.audit_log_id == audit_id).order_by(Mapping.id)
        return list(s.scalars(q).all())
    finally:
        s.close()


def _page_chat() -> None:
    st.subheader("チャット")
    st.caption(
        "メッセージは Ollama で匿名化されたうえで Gemini に送られ、回答はプレースホルダを復元して表示されます。"
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("メッセージを入力…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("匿名化 → Gemini → 復元 中…"):
                try:
                    result = process_gateway(prompt)
                    st.markdown(result.output_restored)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": result.output_restored}
                    )
                    with st.expander("このリクエストのパイプライン詳細（ログ形式）", expanded=False):
                        report = format_pipeline_report(
                            audit_id=result.audit_id,
                            status="restored",
                            created_at=result.created_at,
                            input_raw=result.input_raw,
                            text_anonymized=result.text_anonymized,
                            text_gemini_raw=result.text_gemini_raw,
                            output_restored=result.output_restored,
                            include_gemini=True,
                        )
                        st.code(report, language=None)
                except Exception as e:
                    st.error(f"エラー: {e}")
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": f"（エラー）{e}"}
                    )
                    with st.expander("トレースバック"):
                        st.code(traceback.format_exc(), language="python")

    if st.button("チャット履歴をクリア", type="secondary"):
        st.session_state.chat_messages = []
        st.rerun()


def _page_db() -> None:
    st.subheader("DB 確認")
    st.caption("`audit_logs` の一覧と、監査 ID ごとの `mappings` を表示します。")

    rows = _list_audits()
    if not rows:
        st.info("まだ監査ログがありません。チャットで 1 件以上実行してください。")
        return

    st.markdown("##### 監査ログ（`audit_logs`）")
    table = [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "status": r.status,
            "input_preview": _snippet(r.input_raw, 64),
            "anon_preview": _snippet(r.text_anonymized, 64),
            "out_preview": _snippet(r.output_restored, 64),
        }
        for r in rows
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Mapping（`mappings`）")
    ids = [r.id for r in rows]
    aid = st.selectbox(
        "監査 ID を選択",
        options=ids,
        index=0,
        format_func=lambda i: f"#{i}",
    )
    maps = _list_mappings(int(aid))
    if not maps:
        st.warning("この監査 ID に紐づく Mapping 行がありません。")
    else:
        st.dataframe(
            [
                {
                    "id": m.id,
                    "entity_kind": m.entity_kind,
                    "original_value": m.original_value,
                    "placeholder": m.placeholder,
                }
                for m in maps
            ],
            use_container_width=True,
            hide_index=True,
        )


def _page_pipeline() -> None:
    st.subheader("パイプライン一括確認")
    st.caption("元データ・匿名化データ・Gemini 応答・最終出力を、CLI の `--show-pipeline` と同じ形式でまとめて表示します。")

    rows = _list_audits()
    if not rows:
        st.info("表示する監査ログがありません。")
        return

    labels = [
        f"#{r.id}  {r.status}  {_snippet(r.input_raw, 40)}"
        for r in rows
    ]
    idx = st.selectbox(
        "監査ログを選択",
        options=list(range(len(rows))),
        format_func=lambda i: labels[i],
        index=0,
    )
    selected_id = rows[int(idx)].id
    include_gemini = st.checkbox("Gemini 応答ブロックを含める", value=True)

    fresh = _get_audit(selected_id)
    if fresh is None:
        st.error("行が見つかりません。")
        return

    report = format_pipeline_report(
        audit_id=fresh.id,
        status=fresh.status,
        created_at=fresh.created_at,
        input_raw=fresh.input_raw,
        text_anonymized=fresh.text_anonymized,
        text_gemini_raw=fresh.text_gemini_raw,
        output_restored=fresh.output_restored,
        include_gemini=include_gemini,
    )
    st.code(report, language=None)


def main() -> None:
    st.set_page_config(
        page_title="A.G.I.S.",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _bootstrap()

    st.sidebar.title("A.G.I.S.")
    st.sidebar.markdown("Anonymous Gateway Inference System")
    page = st.sidebar.radio(
        "ページ",
        ("チャット", "DB 確認", "パイプライン一括"),
        index=0,
    )

    st.title("A.G.I.S. コンソール")

    if page == "チャット":
        _page_chat()
    elif page == "DB 確認":
        _page_db()
    else:
        _page_pipeline()


if __name__ == "__main__":
    main()
