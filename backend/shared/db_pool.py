"""Thread-local SQLite connection pool.

Provides a single, stable place to acquire SQLite connections. Every thread
gets its own cached connection (thread-local storage), which:
  - Eliminates the per-request connection open/close overhead.
  - Is safe without locks for reads (each thread owns its connection).
  - Is still safe for writes because SQLite's WAL journal handles concurrency
    at the file level.

Usage
-----
    from backend.shared.db_pool import ThreadLocalPool
    pool = ThreadLocalPool("controlplane.db")
    conn = pool.get_conn()
    conn.execute(...)

Or via the module-level helper for the default DB path:
    from backend.shared.db_pool import get_conn
    conn = get_conn(path)

Future swap
-----------
Replace ``_make_conn`` with a psycopg2 / asyncpg factory to switch to
PostgreSQL without touching any caller code.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class ThreadLocalPool:
    """One SQLite connection per thread, configured for WAL mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._conn_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_conn(self) -> sqlite3.Connection:
        """Open a new connection and apply WAL + performance PRAGMAs."""
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,  # we manage thread-safety ourselves
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        with self._conn_lock:
            conn.execute("PRAGMA busy_timeout=5000")   # wait 5 s before LOCKED error
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
            conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_conn(self) -> sqlite3.Connection:
        """Return the current thread's cached connection, opening one if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_conn()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the current thread's connection (useful in tests / cleanup)."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


# ---------------------------------------------------------------------------
# Module-level singleton pools keyed by absolute path
# ---------------------------------------------------------------------------
_pools: dict[str, ThreadLocalPool] = {}
_pools_lock = threading.Lock()


def get_pool(path: str | Path) -> ThreadLocalPool:
    """Return (or create) the singleton ThreadLocalPool for *path*."""
    abs_path = str(Path(path).resolve())
    with _pools_lock:
        if abs_path not in _pools:
            _pools[abs_path] = ThreadLocalPool(abs_path)
        return _pools[abs_path]


def get_conn(path: str | Path) -> sqlite3.Connection:
    """Convenience function: get the thread-local connection for *path*."""
    return get_pool(path).get_conn()
