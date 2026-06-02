# A.G.I.S.（Anonymous Gateway Inference System）

ローカル LLM（Ollama）が **文章内容** を解析してマスキングし、選択した **業界の秘匿ルールパック** で抽出方針を最適化します。既定の **抽象化マスキング**（例: `34歳`→`30代前半`、`35歳`→`30代後半`、`群馬県`→`関東圏`、個人名→`患者1`）で文意を保ったまま秘匿し、**抽象化したテキストだけ**を Gemini API に送ります。GUI では応答を「抽象のまま」または「原文へ復元」で切り替え表示できます。**CLI** と **Streamlit GUI** の両方から利用できます。

管理者は任意で正規表現ルールを追加できます（`MASKING_SOURCE=both` 時のみ上書き適用）。日常運用では業界設定と自動抽出のみで足ります。

SQLite（`data/`）には **`AuditLog`**（1 リクエストの入出力・業界）、**`Mapping`**（置換対応）、**`MaskingRule`**（管理者向けルール）、**`ActivityLog`**（操作ログ）が保存されます。

## 技術スタック

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.11+（Dockerfile は 3.11-slim） |
| ローカル LLM | Ollama（既定モデル: `gemma3:4b`） |
| 外部 API | Google Gemini（`google-generativeai`、既定: `gemini-2.5-flash`） |
| DB | SQLite + SQLAlchemy 2.0 |
| コンテナ | Docker / Docker Compose（任意） |
| GUI | Streamlit（`gui/streamlit_app.py`） |

## リポジトリ構成（主要ファイル）

- `app/main.py` — ゲートウェイ本体・CLI（`process_gateway`）
- `app/anonymizer.py` — 業界連動 Ollama 抽出と置換・復元
- `app/masking_abstract.py` — ジャンル別抽象化（年齢・住所・連番など）
- `app/masking_policy.py` — 業界パック読込・抽出プロンプト生成
- `data/abstraction_rules.json` — 抽象化戦略・都道府県→地方ブロック
- `app/models.py` — `AuditLog` / `Mapping` / `MaskingRule` / `ActivityLog`
- `app/masking_genres.py` — 秘匿ジャンル（年齢・自社名など）の定義
- `app/rule_masking.py` — 管理者ルールの読込・マージ（`both` 時）
- `data/industries.json` — 業界一覧
- `data/industry_packs/*.json` — 業界別マスキング方針
- `data/masking_genres.json` — 秘匿ジャンル定義（GUI からも編集可）
- `gui/streamlit_app.py` — **Streamlit GUI**
- `.env.example` — 環境変数テンプレート

## 前提条件

- **Ollama** が利用可能なホストで動作（例: `ollama pull gemma3:4b`）
- **Gemini API キー**（[Google AI Studio](https://aistudio.google.com/) 等）

## セットアップ（ホストで直接実行）

```bash
cd /path/to/A.G.I.S.-Anonymous-Gateway-Inference-System-
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# .env を編集し GEMINI_API_KEY を設定する
```

実行時はプロジェクトルートで `PYTHONPATH` を通します。

```bash
export PYTHONPATH=.
# ホストから Ollama へは通常 127.0.0.1（.env.example 参照）
export DATABASE_URL=sqlite:///./data/agis.db   # 未設定なら .env の値を利用
```

## 環境変数（抜粋）

| 変数 | 説明 |
|------|------|
| `MASKING_SOURCE` | `auto`（既定・業界連動 Ollama）/ `both`（+ 管理者ルール）/ `rules`（非推奨）。`ollama` は `auto` のエイリアス。 |
| `MASKING_MODE` | `abstract`（既定・文意保持の抽象化）/ `placeholder`（`[PERSON_1]` 形式・後方互換）。 |
| `AGIS_INDUSTRY` | 業界 ID（`general` / `healthcare` / `finance` / `legal_hr`）。GUI・CLI で上書き可。 |
| `AGIS_ABSTRACTION_RULES_PATH` | 抽象化ルール JSON（省略時 `data/abstraction_rules.json`）。 |
| `MASKING_GENRES_PATH` | 秘匿ジャンル JSON（省略時 `data/masking_genres.json`）。 |
| `AGIS_INDUSTRIES_PATH` | 業界一覧 JSON（省略時 `data/industries.json`）。 |
| `AGIS_INDUSTRY_PACKS_DIR` | 業界パックディレクトリ（省略時 `data/industry_packs/`）。 |
| `AGIS_ACTOR` | 操作ログ・CLI で actor 未指定時の既定（空なら `anonymous`）。 |
| `GEMINI_API_KEY` | Gemini 呼び出しに必須。 |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Ollama（`auto` または `both` のとき）。 |

## マスキングの流れ

1. **業界**を設定（`.env` の `AGIS_INDUSTRY`、GUI サイドバー、CLI `--industry`）
2. Ollama が文章を読み、業界パックの重点ジャンル＋文脈上の秘匿情報を抽出
3. ジャンル別ルールで **抽象化置換**（年齢→`30代前半`/`30代後半`、住所→地方、個人名→`患者1` など）し Gemini へ送信
4. 応答は GUI で「抽象のまま」または Mapping に基づく「原文復元」を選択表示

`MASKING_MODE=placeholder` にすると従来の `[ジャンル_n]` 置換になります。

業界パックを追加するには `data/industry_packs/` に JSON を置き、`data/industries.json` にエントリを追加します。

## GUI（Streamlit）

```bash
cd /path/to/A.G.I.S.-Anonymous-Gateway-Inference-System-
export PYTHONPATH=.
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run gui/streamlit_app.py
```

- **チャット**: サイドバーで **業界**・**応答表示（抽象/復元）**・Actor を設定。同一セッション内の過去のやり取り（匿名化済み）を Gemini に渡すため、続きの質問も文脈を維持します。「チャット履歴をクリア」で文脈をリセット。
- **DB 確認**: `audit_logs`（業界列含む）と `mappings`。
- **パイプライン一括**: 保存済み監査ログの一括表示。
- **管理者ルール**: 任意の正規表現上書き（`MASKING_SOURCE=both` 時）。ジャンル JSON の編集もここから可能。
- **操作ログ**: `gateway.*` / `rule.*` / `genre.*` など。

## CLI の使い方

```bash
.venv/bin/python -m app.main run -m "患者ID 12345、診断: 2型糖尿病" --industry healthcare
.venv/bin/python -m app.main run -m "山田太郎に連絡してください" --actor "yamada@corp"
.venv/bin/python -m app.main run -m "..." --show-pipeline
.venv/bin/python -m app.main show --last
```

## Docker で動かす

```bash
docker compose build
docker compose run --rm app python -m app.main run -m "テスト文" --industry general
```

ホストで直接実行する場合は `.env` の `OLLAMA_BASE_URL` を `http://127.0.0.1:11434` にしてください。

## 開発者向けメモ

- **`process_gateway(user_text, *, actor=None, industry=None)`** — `industry` 未指定時は `AGIS_INDUSTRY` → `general`。
- 監査ログに `industry` 列が保存されます（既存 DB は `init_db()` でマイグレーション）。

## トラブルシューティング

| 現象 | 対処の例 |
|------|-----------|
| Ollama 接続エラー | ホスト実行時は `OLLAMA_BASE_URL=http://127.0.0.1:11434`。 |
| `404` モデル未検出 | `GEMINI_MODEL` を利用可能な ID に変更。 |
| `429` クォータ超過 | 時間を空ける、別モデル・課金設定の確認。 |
| `MASKING_SOURCE=rules` でルール 0 件 | 原文のまま送られる可能性あり。`auto` に変更するかルールを追加。 |

## ライセンス・注意

本リポジトリは実験・開発用の構成です。個人情報の取り扱いは利用環境のポリシーと法令に従ってください。
