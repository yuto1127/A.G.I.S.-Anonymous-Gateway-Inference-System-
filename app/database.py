"""Database engine, session factory, and schema initialization."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCKER_DB_URL = "sqlite:////app/data/agis.db"

# Load .env from repo root so Streamlit/CLI cwd does not matter.
load_dotenv(_REPO_ROOT / ".env", override=False)

from app import models as _models  # noqa: F401 — register ORM tables on Base.metadata
from app.models import Base


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _host_default_db_url() -> str:
    """Absolute SQLite URL under `<repo>/data/agis.db`."""
    db_path = _REPO_ROOT / "data" / "agis.db"
    return f"sqlite:///{db_path.as_posix()}"


def get_database_url() -> str:
    """Resolve SQLite URL: env > host default; rewrite Docker URL when not in container."""
    raw = os.environ.get("DATABASE_URL", "").strip()
    if not raw:
        return _DOCKER_DB_URL if _running_in_docker() else _host_default_db_url()
    if raw == _DOCKER_DB_URL and not _running_in_docker():
        return _host_default_db_url()
    if raw.startswith("sqlite:///") and not _running_in_docker():
        path_part = raw[len("sqlite:///") :]
        if path_part and path_part != ":memory:" and not path_part.startswith("/"):
            resolved = (_REPO_ROOT / path_part).resolve()
            return f"sqlite:///{resolved.as_posix()}"
    return raw


def _ensure_sqlite_directory(url: str) -> None:
    """Create parent directory for file-based SQLite URLs."""
    if not url.startswith("sqlite"):
        return
    # sqlite:////absolute/path (4 slashes) or sqlite:///relative
    path_part = url.split("sqlite:///", 1)[-1] if "sqlite:///" in url else ""
    if not path_part or path_part == ":memory:":
        return
    if path_part.startswith("/"):
        db_file = Path(path_part)
    else:
        db_file = Path(path_part)
        if not db_file.is_absolute():
            db_file = Path.cwd() / db_file
    db_file.parent.mkdir(parents=True, exist_ok=True)


def create_engine_from_env():
    """Create SQLAlchemy engine; `check_same_thread=False` for SQLite + threaded httpx."""
    url = get_database_url()
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args)


engine = create_engine_from_env()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def _ensure_audit_logs_industry_column() -> None:
    """SQLite: add `industry` to legacy audit_logs tables."""
    if not get_database_url().startswith("sqlite"):
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info('audit_logs')")).fetchall()
        col_names = [r[1] for r in rows]
        if "industry" not in col_names:
            conn.execute(
                text(
                    "ALTER TABLE audit_logs ADD COLUMN industry VARCHAR(64) NOT NULL DEFAULT 'general'"
                )
            )


def _ensure_masking_rules_genre_column() -> None:
    """SQLite: add `genre` to legacy DBs created before the column existed."""
    if not get_database_url().startswith("sqlite"):
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info('masking_rules')")).fetchall()
        col_names = [r[1] for r in rows]
        if "genre" not in col_names:
            conn.execute(
                text(
                    "ALTER TABLE masking_rules ADD COLUMN genre VARCHAR(64) NOT NULL DEFAULT 'OTHER'"
                )
            )


def init_db() -> None:
    """Create all tables if they do not exist; apply lightweight SQLite patches."""
    _ensure_sqlite_directory(get_database_url())
    Base.metadata.create_all(bind=engine)
    _ensure_audit_logs_industry_column()
    _ensure_masking_rules_genre_column()
    from app.masking_genres import ensure_genres_file

    ensure_genres_file()


@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
