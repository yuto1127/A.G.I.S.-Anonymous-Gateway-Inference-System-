# A.G.I.S.（Anonymous Gateway Inference System）

ローカル LLM（Ollama）で個人情報を検出・プレースホルダ化し、**匿名化したテキストだけ**を Gemini API に送り、返答を DB に保存したマッピングで復元するゲートウェイです。**CLI** と **Streamlit GUI** の両方から利用できます。処理の各段階は SQLite（`data/`）の **`AuditLog`** と **`Mapping`** に記録されます。

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
- `app/anonymizer.py` — Ollama による抽出 JSON のパースと置換・復元ヘルパ
- `app/models.py` — `AuditLog` / `Mapping`
- `app/database.py` — エンジン・セッション・`init_db()`
- `gui/streamlit_app.py` — **チャット形式 GUI**（DB 確認・パイプライン一括表示を同梱）
- `Dockerfile` / `docker-compose.yml` — アプリ用コンテナと `data` マウント
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

## GUI（Streamlit）

ブラウザでチャットしつつ、**DB（監査ログ・Mapping）**と**元／匿名化／Gemini／復元の一括ログ**を同じアプリ内のページで確認できます。

```bash
cd /path/to/A.G.I.S.-Anonymous-Gateway-Inference-System-
export PYTHONPATH=.
.venv/bin/pip install -r requirements.txt   # streamlit を含む
.venv/bin/streamlit run gui/streamlit_app.py
```

- **チャット**: `st.chat_input` で送信。各応答の下のエキスパンダに、当該リクエストの `format_pipeline_report` 相当の詳細を表示します。
- **DB 確認**: 直近の `audit_logs` 一覧と、監査 ID を選んだときの `mappings` 表です。
- **パイプライン一括**: 保存済み行を選び、CLI の `--show-pipeline` と同形式のテキストを表示します（Gemini ブロックの ON/OFF 可）。

`.env`（`GEMINI_API_KEY`、`OLLAMA_BASE_URL` 等）はリポジトリルートをカレントにして起動すると読み込まれます。

## CLI の使い方

### ゲートウェイを 1 回実行する

```bash
.venv/bin/python -m app.main -m "山田太郎に連絡してください"
# または明示的に
.venv/bin/python -m app.main run -m "山田太郎に連絡してください"
```

### 元データ・匿名化・Gemini 応答・復元をまとめて表示

```bash
.venv/bin/python -m app.main -m "山田太郎に連絡してください" --show-pipeline
```

Gemini 応答ブロックだけ省略する場合:

```bash
.venv/bin/python -m app.main run -m "..." --show-pipeline --no-gemini-block
```

### 保存済み監査ログを表示（API を再呼び出ししない）

```bash
.venv/bin/python -m app.main show --last
.venv/bin/python -m app.main show --id 3
```

## Docker で動かす

```bash
docker compose build
# .env に GEMINI_API_KEY を記入したうえで
docker compose run --rm app python -m app.main run -m "テスト文"
```

Compose 側で `OLLAMA_BASE_URL` が `host.docker.internal` 向きに設定されている場合、**コンテナ内**からホストの Ollama に接続します。ホストで直接 `python -m app.main` する場合は `.env` の `OLLAMA_BASE_URL` を `http://127.0.0.1:11434` にしてください。

## 開発者向けメモ

- **`process_gateway(user_text: str)`** の戻り値は **`GatewayResult`**（`audit_id`・各段階の文字列など）です。最終テキストのみ必要な場合は `result.output_restored` を参照してください。
- `google.generativeai` 利用時に **FutureWarning** が出る場合があります（非推奨パッケージの案内）。動作自体とは別問題です。

## トラブルシューティング（よくあるもの）

| 現象 | 対処の例 |
|------|-----------|
| Ollama 接続エラー（ホスト名解決など） | ホスト実行時は `OLLAMA_BASE_URL=http://127.0.0.1:11434`。コンテナ外では `host.docker.internal` が使えないことがあります（アプリ側でコンテナ外時の置換にも対応済みの場合があります）。 |
| `404` モデル未検出 | `GEMINI_MODEL` を `gemini-2.5-flash` 等、現行で利用可能なモデル ID に変更。 |
| `429` クォータ超過 | 無料枠の上限・別モデルの枠の差があります。時間を空ける、[レート制限の案内](https://ai.google.dev/gemini-api/docs/rate-limits)の確認、必要に応じて課金設定の見直し。 |

## ライセンス・注意

本リポジトリは実験・開発用の構成です。個人情報の取り扱いは利用環境のポリシーと法令に従ってください。
