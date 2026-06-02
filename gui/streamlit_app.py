"""A.G.I.S. ローカル Web GUI（Streamlit）: チャット・DB・ルール・操作ログ。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select

from app.activity import write_activity
from app.actor import resolve_actor
from app.database import SessionLocal, init_db
from app.main import format_pipeline_report, process_gateway
from app.masking_policy import list_industries, resolve_industry
from app.masking_genres import (
    genre_label,
    genre_options_tuples,
    get_genres_path,
    load_genres_list,
    save_genres_list,
    validate_genre_entry,
)
from app.models import ActivityLog, AuditLog, Mapping, MaskingRule
from app.rule_masking import (
    InvalidMaskingPatternError,
    extract_spans_from_rules,
    validate_masking_pattern,
)


def _bootstrap() -> None:
    load_dotenv(_ROOT / ".env", override=False)
    init_db()


def _actor_for_requests() -> str | None:
    """Explicit actor from sidebar; None lets resolve_actor use AGIS_ACTOR."""
    v = st.session_state.get("actor_gui_field", "")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


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


def _list_masking_rules() -> list[MaskingRule]:
    s = SessionLocal()
    try:
        q = select(MaskingRule).order_by(MaskingRule.priority.desc(), MaskingRule.id.desc())
        return list(s.scalars(q).all())
    finally:
        s.close()


def _list_activity_logs(limit: int = 400) -> list[ActivityLog]:
    s = SessionLocal()
    try:
        q = select(ActivityLog).order_by(ActivityLog.id.desc()).limit(limit)
        return list(s.scalars(q).all())
    finally:
        s.close()


def _selected_industry() -> str:
    """Industry from sidebar session state."""
    return resolve_industry(st.session_state.get("industry_gui"))


def _display_mode() -> str:
    """abstract | restored — how assistant messages are shown."""
    return st.session_state.get("display_mode", "abstract")


def _assistant_content(msg: dict) -> str:
    if _display_mode() == "restored":
        return msg.get("content_restored") or msg.get("content", "")
    return msg.get("content_abstract") or msg.get("content", "")


def _page_chat() -> None:
    st.subheader("チャット")
    src = os.environ.get("MASKING_SOURCE", "auto")
    ind = _selected_industry()
    ind_label = next((lab for i, lab in list_industries() if i == ind), ind)
    mode = os.environ.get("MASKING_MODE", "abstract")
    st.caption(
        f"業界: **{ind_label}** (`{ind}`) ／ マスキング: **{mode}** ／ ソース: **{src}**。"
        "抽象化マスキングで文意を保ちつつ秘匿します（例: 34歳→30代前半、35歳→30代後半、群馬県→関東圏）。"
        "同一チャット内のやり取りは匿名化済み履歴として Gemini に渡されます。"
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_history_anon" not in st.session_state:
        st.session_state.chat_history_anon = []
    if "chat_audit_ids" not in st.session_state:
        st.session_state.chat_audit_ids = []

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            text = _assistant_content(m) if m["role"] == "assistant" else m.get("content", "")
            st.markdown(text)

    if prompt := st.chat_input("メッセージを入力…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("匿名化 → Gemini → 復元 中…"):
                try:
                    result = process_gateway(
                        prompt,
                        actor=_actor_for_requests(),
                        industry=_selected_industry(),
                        gemini_history=list(st.session_state.chat_history_anon),
                        conversation_audit_ids=list(st.session_state.chat_audit_ids),
                    )
                    shown = (
                        result.output_restored
                        if _display_mode() == "restored"
                        else result.output_abstract
                    )
                    st.markdown(shown)
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content_restored": result.output_restored,
                            "content_abstract": result.output_abstract,
                        }
                    )
                    st.session_state.chat_history_anon.append(
                        {"role": "user", "content": result.text_anonymized}
                    )
                    st.session_state.chat_history_anon.append(
                        {"role": "model", "content": result.text_gemini_raw}
                    )
                    st.session_state.chat_audit_ids.append(result.audit_id)
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
        st.session_state.chat_history_anon = []
        st.session_state.chat_audit_ids = []
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
            "industry": getattr(r, "industry", None) or "general",
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
    st.caption("元データ・匿名化データ・Gemini 応答・最終出力をまとめて表示します。")

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


def _render_genre_editor() -> None:
    """CRUD for `data/masking_genres.json` (or MASKING_GENRES_PATH)."""
    st.markdown("##### ジャンル一覧（JSON）")
    path = get_genres_path()
    st.caption(f"ファイル: `{path}` ・ 環境変数 `MASKING_GENRES_PATH` で別パスを指定できます。")

    try:
        genres = load_genres_list()
    except (OSError, json.JSONDecodeError, ValueError) as e:
        st.error(f"読み込みエラー: {e}")
        return

    st.dataframe(genres, use_container_width=True, hide_index=True)

    st.markdown("##### ジャンルを追加")
    with st.form("genre_add"):
        gc = st.text_input("コード（例: PROJECT_CODE）", placeholder="PROJECT_CODE")
        gl = st.text_input("ラベル（表示名）", placeholder="プロジェクトコード")
        if st.form_submit_button("追加"):
            try:
                validate_genre_entry(gc.strip(), gl.strip())
            except ValueError as e:
                st.error(str(e))
            else:
                ncode, nlab = gc.strip(), gl.strip()
                if any(x["code"] == ncode for x in genres):
                    st.error("同じコードのジャンルが既にあります。")
                else:
                    try:
                        save_genres_list(genres + [{"code": ncode, "label": nlab}])
                        write_activity(
                            actor=resolve_actor(_actor_for_requests()),
                            action="genre.create",
                            summary=f"genre {ncode}",
                            detail={"code": ncode},
                        )
                    except ValueError as e2:
                        st.error(str(e2))
                    else:
                        st.success("追加しました。")
                        st.rerun()

    st.markdown("##### ラベルを編集")
    codes = [x["code"] for x in genres]
    if codes:
        pick = st.selectbox(
            "編集するコード",
            options=codes,
            format_func=lambda c: f"{c} — {genre_label(c)}",
            key="genre_edit_pick",
        )
        cur = next(x for x in genres if x["code"] == pick)
        with st.form("genre_edit"):
            new_label = st.text_input("新しいラベル", value=cur["label"])
            if st.form_submit_button("ラベルを保存"):
                try:
                    validate_genre_entry(pick, new_label.strip())
                except ValueError as e:
                    st.error(str(e))
                else:
                    updated = [
                        {"code": x["code"], "label": new_label.strip() if x["code"] == pick else x["label"]}
                        for x in genres
                    ]
                    try:
                        save_genres_list(updated)
                        write_activity(
                            actor=resolve_actor(_actor_for_requests()),
                            action="genre.update",
                            summary=f"genre label {pick}",
                            detail={"code": pick},
                        )
                    except ValueError as e2:
                        st.error(str(e2))
                    else:
                        st.success("更新しました。")
                        st.rerun()

    st.markdown("##### ジャンルを削除")
    del_candidates = [x["code"] for x in genres if x["code"] != "OTHER"]
    if not del_candidates:
        st.caption("`OTHER` 以外のジャンルがありません。")
    else:
        del_code = st.selectbox("削除するコード", options=del_candidates, key="genre_del_pick")
        if st.button("このジャンルを削除", type="primary", key="genre_del_btn"):
            new_list = [x for x in genres if x["code"] != del_code]
            try:
                save_genres_list(new_list)
                write_activity(
                    actor=resolve_actor(_actor_for_requests()),
                    action="genre.delete",
                    summary=f"genre deleted {del_code}",
                    detail={"code": del_code},
                )
            except ValueError as e:
                st.error(str(e))
            else:
                st.success("削除しました。")
                st.rerun()


def _page_rules() -> None:
    st.subheader("管理者向けルール（任意）")
    st.info(
        "通常はサイドバーの **業界** 設定と文章内容による自動マスキングのみで運用します。"
        "下記の正規表現ルールは、`MASKING_SOURCE=both` のときのみ上書きとして適用されます。"
    )
    st.caption(
        "ジャンル定義は JSON ファイルで管理し、この画面からも編集できます。"
    )

    tab_rules, tab_genres = st.tabs(["管理者ルール", "ジャンル（JSON）"])
    with tab_genres:
        _render_genre_editor()
    with tab_rules:
        _render_masking_rules_tab()


def _render_masking_rules_tab() -> None:
    opts = genre_options_tuples()
    genre_codes = [c for c, _ in opts]
    st.caption(
        "ジャンルは分類・表示用で、プレースホルダの既定接頭辞にも使います（接頭辞の上書きは任意）。"
        "優先度は数値が大きいほど高く、重なるマッチは高優先が残ります。"
    )
    rules = _list_masking_rules()
    st.markdown("##### 登録済みルール")
    if rules:
        for r in rules:
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                g = getattr(r, "genre", None) or "OTHER"
                g_lab = genre_label(g)
                ek = (r.entity_kind or "").strip() or f"（ジャンル既定: {g}）"
                st.write(
                    f"**#{r.id}** `{r.name}`  ジャンル: **{g_lab}** (`{g}`)  priority={r.priority}  "
                    f"enabled={'はい' if r.enabled else 'いいえ'}  接頭辞: `{ek}`"
                )
                st.code(r.pattern, language=None)
                if r.description:
                    st.caption(r.description)
            with c2:
                if st.button("無効化" if r.enabled else "有効化", key=f"tg_{r.id}"):
                    s = SessionLocal()
                    try:
                        row = s.get(MaskingRule, r.id)
                        if row:
                            row.enabled = not row.enabled
                            s.commit()
                            write_activity(
                                actor=resolve_actor(_actor_for_requests()),
                                action="rule.toggle",
                                summary=f"rule #{r.id} enabled={row.enabled}",
                                detail={"rule_id": r.id},
                            )
                    finally:
                        s.close()
                    st.rerun()
            with c3:
                if st.button("削除", key=f"dl_{r.id}", type="primary"):
                    s = SessionLocal()
                    try:
                        row = s.get(MaskingRule, r.id)
                        if row:
                            s.delete(row)
                            s.commit()
                            write_activity(
                                actor=resolve_actor(_actor_for_requests()),
                                action="rule.delete",
                                summary=f"rule #{r.id} deleted",
                                detail={"rule_id": r.id},
                            )
                    finally:
                        s.close()
                    st.rerun()
            st.divider()
    else:
        st.info("ルールがまだありません。下のフォームから追加してください。")

    st.markdown("##### ルールを追加")
    with st.form("add_rule"):
        name = st.text_input("名前", placeholder="例: 社内メール")
        genre = st.selectbox(
            "秘匿ジャンル",
            options=genre_codes,
            format_func=lambda c: genre_label(c),
            index=0,
        )
        pattern = st.text_input("正規表現", placeholder=r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
        entity_kind = st.text_input(
            "プレースホルダ接頭辞の上書き（任意）",
            placeholder="空欄ならジャンルコードを使用（例: EMAIL, COMPANY_SELF）",
            help="英大文字推奨。空欄のときは上で選んだジャンルコードが [ジャンル_1] の接頭辞になります。",
        )
        priority = st.number_input("優先度（大きいほど強い）", value=10, step=1)
        description = st.text_area("説明（任意）", height=68)
        enabled = st.checkbox("有効", value=True)
        submitted = st.form_submit_button("追加")
        if submitted:
            try:
                validate_masking_pattern(pattern)
            except InvalidMaskingPatternError as e:
                st.error(f"正規表現エラー: {e}")
            else:
                s = SessionLocal()
                try:
                    row = MaskingRule(
                        name=name or "unnamed",
                        pattern=pattern,
                        genre=genre,
                        entity_kind=entity_kind.strip(),
                        priority=int(priority),
                        description=description or None,
                        enabled=enabled,
                    )
                    s.add(row)
                    s.commit()
                    s.refresh(row)
                    write_activity(
                        actor=resolve_actor(_actor_for_requests()),
                        action="rule.create",
                        summary=f"rule #{row.id} {row.name}",
                        detail={
                            "rule_id": row.id,
                            "pattern_len": len(pattern),
                            "genre": genre,
                        },
                    )
                finally:
                    s.close()
                st.success("追加しました。")
                st.rerun()

    st.markdown("##### プレビュー（現在のルールのみマッチ箇所）")
    sample = st.text_area("サンプル文", height=100, key="rule_preview_sample")
    if st.button("プレビュー実行", key="rule_preview_btn"):
        s = SessionLocal()
        try:
            spans = extract_spans_from_rules(s, sample)
        finally:
            s.close()
        st.json(
            [
                {"start": sp.start, "end": sp.end, "value": sp.value, "kind": sp.kind}
                for sp in spans
            ]
        )


def _page_activity() -> None:
    st.subheader("操作ログ（activity_logs）")
    st.caption("`gateway.*` / `rule.*` / `genre.*` など、誰がどの操作をしたかの記録です。")

    rows = _list_activity_logs()
    if not rows:
        st.info("まだ操作ログがありません。")
        return
    st.dataframe(
        [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "actor": r.actor,
                "action": r.action,
                "summary": _snippet(r.summary, 120),
                "audit_log_id": r.audit_log_id,
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("##### 詳細（JSON）")
    pick = st.selectbox("行を選択", options=[r.id for r in rows], format_func=lambda i: f"#{i}")
    row = next(x for x in rows if x.id == pick)
    st.text_area("detail_json", value=row.detail_json or "", height=160, disabled=True)


def main() -> None:
    st.set_page_config(
        page_title="A.G.I.S.",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _bootstrap()

    if "actor_gui_field" not in st.session_state:
        st.session_state.actor_gui_field = os.environ.get("AGIS_ACTOR", "")

    industry_opts = list_industries()
    industry_ids = [i for i, _ in industry_opts]
    default_industry = resolve_industry(None)
    if "industry_gui" not in st.session_state:
        st.session_state.industry_gui = (
            default_industry if default_industry in industry_ids else industry_ids[0]
        )

    st.sidebar.title("A.G.I.S.")
    st.sidebar.markdown("Anonymous Gateway Inference System")
    st.sidebar.selectbox(
        "業界（マスキング最適化）",
        options=industry_ids,
        format_func=lambda i: next(lab for x, lab in industry_opts if x == i),
        key="industry_gui",
        help="文章内容の解析に加え、選択した業界の秘匿ルールパックを適用します。",
    )
    st.sidebar.caption(f"実効業界: **{resolve_industry(st.session_state.industry_gui)}**")
    st.sidebar.text_input(
        "Actor（操作ログ用）",
        key="actor_gui_field",
        help="空欄のときは環境変数 AGIS_ACTOR、なければ anonymous。",
    )
    st.sidebar.caption(f"実効 actor: **{resolve_actor(_actor_for_requests())}**")
    st.sidebar.caption(
        f"マスキング: `{os.environ.get('MASKING_MODE', 'abstract')}` ／ "
        f"ソース: `{os.environ.get('MASKING_SOURCE', 'auto')}`"
    )
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = "abstract"
    st.sidebar.radio(
        "応答の表示",
        options=("abstract", "restored"),
        format_func=lambda x: "抽象のまま" if x == "abstract" else "復元（原文へ置換）",
        key="display_mode",
        help="Gemini の応答を、抽象化テキストのまま見るか、Mapping に基づき原文語に戻すか。",
    )

    page = st.sidebar.radio(
        "ページ",
        ("チャット", "DB 確認", "パイプライン一括", "管理者ルール", "操作ログ"),
        index=0,
    )

    st.title("A.G.I.S. コンソール")

    if page == "チャット":
        _page_chat()
    elif page == "DB 確認":
        _page_db()
    elif page == "パイプライン一括":
        _page_pipeline()
    elif page == "管理者ルール":
        _page_rules()
    else:
        _page_activity()


if __name__ == "__main__":
    main()
