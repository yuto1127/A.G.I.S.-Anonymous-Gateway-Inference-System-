"""Database engine, session factory, and schema initialization."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

from app import models as _models  # noqa: F401 — register ORM tables on Base.metadata
from app.models import Base

# Default matches Docker compose volume mount.
_DEFAULT_DB_URL = "sqlite:////app/data/agis.db"


def get_database_url() -> str:
    """Resolve SQLite URL from environment (Docker or local)."""
    return os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)


def create_engine_from_env():
    """Create SQLAlchemy engine; `check_same_thread=False` for SQLite + threaded httpx."""
    url = get_database_url()
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args)


engine = create_engine_from_env()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


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
