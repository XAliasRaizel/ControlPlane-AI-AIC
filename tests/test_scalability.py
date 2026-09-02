"""Scalability tests for ControlPlane.ai backend.

Covers:
- WAL mode is enabled on Database init
- ThreadLocalPool returns the same connection on the same thread
- ThreadLocalPool returns different connections on different threads
- SQLiteSessionStore: session survives re-opening the same DB (process-restart sim)
- SQLiteSessionStore: vacuum_sessions removes stale rows correctly
- TTLCache: returns cached value within TTL
- TTLCache: returns None after TTL expiry
- TTLCache: invalidate clears a cached entry
- AsyncTaskQueue: enqueued coroutines are executed
- AsyncTaskQueue: back-pressure when full (no block, just warn + drop)
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import sqlite3
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db(tmp_path):
    """Return an absolute path to a temporary SQLite file."""
    return str(tmp_path / "test_scalability.db")


# ---------------------------------------------------------------------------
# 1. WAL mode tests (ThreadLocalPool + Database)
# ---------------------------------------------------------------------------
class TestWALMode:

    def test_threadlocal_pool_enables_wal(self, tmp_db):
        """ThreadLocalPool should configure WAL journal mode on first connect."""
        from backend.shared.db_pool import ThreadLocalPool
        pool = ThreadLocalPool(tmp_db)
        conn = pool.get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"Expected WAL mode, got: {mode}"

    def test_database_init_enables_wal(self, tmp_db):
        """Database.__init__ should configure WAL via ThreadLocalPool."""
        # Import fresh (pool keyed by absolute path, so tmp_db is unique)
        from backend.shared.db_pool import get_pool
        from pathlib import Path
        pool = get_pool(tmp_db)
        conn = pool.get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout_is_set(self, tmp_db):
        from backend.shared.db_pool import ThreadLocalPool
        pool = ThreadLocalPool(tmp_db)
        conn = pool.get_conn()
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000

    def test_database_class_wal(self, tmp_db):
        """Database (audit/store.py) should expose WAL on its thread-local connection."""
        from backend.audit.store import Database
        db = Database(tmp_db)
        conn = db.connect()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# 2. ThreadLocalPool — same connection per thread, different across threads
# ---------------------------------------------------------------------------
class TestThreadLocalPool:

    def test_same_connection_on_same_thread(self, tmp_db):
        from backend.shared.db_pool import ThreadLocalPool
        pool = ThreadLocalPool(tmp_db)
        conn1 = pool.get_conn()
        conn2 = pool.get_conn()
        assert conn1 is conn2, "Expected the same connection object on the same thread"

    def test_different_connections_on_different_threads(self, tmp_db):
        from backend.shared.db_pool import ThreadLocalPool
        pool = ThreadLocalPool(tmp_db)
        connections = {}

        def grab_conn(name):
            connections[name] = pool.get_conn()

        t1 = threading.Thread(target=grab_conn, args=("t1",))
        t2 = threading.Thread(target=grab_conn, args=("t2",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert connections["t1"] is not connections["t2"], \
            "Each thread should get its own connection"

    def test_connection_works_concurrently(self, tmp_db):
        """Two threads should be able to read concurrently under WAL without error."""
        from backend.shared.db_pool import ThreadLocalPool
        pool = ThreadLocalPool(tmp_db)
        # Create a table first
        pool.get_conn().execute("CREATE TABLE IF NOT EXISTS t(v TEXT)")
        pool.get_conn().commit()

        errors = []

        def read(name):
            try:
                pool.get_conn().execute("SELECT * FROM t").fetchall()
            except Exception as exc:
                errors.append((name, str(exc)))

        threads = [threading.Thread(target=read, args=(f"t{i}",)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Concurrent read errors: {errors}"


# ---------------------------------------------------------------------------
# 3. SQLiteSessionStore — persistence across reopen (restart simulation)
# ---------------------------------------------------------------------------
class TestSQLiteSessionStore:

    def test_session_survives_store_reopen(self, tmp_db):
        """Simulate process restart: write with one store instance, read with another."""
        from backend.risk.session_store import SQLiteSessionStore, SessionState

        store1 = SQLiteSessionStore(tmp_db)
        state = SessionState(
            session_id="sess-abc",
            created_at=time.time(),
            last_updated_at=time.time(),
            ewma_score=0.42,
            turn_count=3,
        )
        store1.set("sess-abc", state, ttl_seconds=3600)

        # Open a brand-new store (different Python object, same DB file)
        # Simulate process restart by clearing the db_pool cache for this path
        from backend.shared import db_pool as _pool_mod
        from pathlib import Path
        abs_path = str(Path(tmp_db).resolve())
        _pool_mod._pools.pop(abs_path, None)

        store2 = SQLiteSessionStore(tmp_db)
        recovered = store2.get("sess-abc")

        assert recovered is not None, "Session should survive DB reopen"
        assert abs(recovered.ewma_score - 0.42) < 1e-9
        assert recovered.turn_count == 3

    def test_vacuum_removes_stale_sessions(self, tmp_db):
        from backend.risk.session_store import SQLiteSessionStore, SessionState
        from datetime import datetime, timedelta, timezone

        store = SQLiteSessionStore(tmp_db, max_age_hours=24)
        state = SessionState(
            session_id="old-sess",
            created_at=time.time(),
            last_updated_at=time.time(),
        )
        store.set("old-sess", state, ttl_seconds=9999)

        # Manually back-date last_updated to 48 hours ago
        conn = store._pool.get_conn()
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        conn.execute("UPDATE sessions SET last_updated=? WHERE session_id=?", (old_ts, "old-sess"))
        conn.commit()

        removed = store.vacuum_sessions(max_age_hours=24)
        assert removed >= 1, f"Expected at least 1 row removed, got {removed}"
        assert store.get("old-sess") is None, "Stale session should be vacuumed"

    def test_session_get_returns_none_for_missing(self, tmp_db):
        from backend.risk.session_store import SQLiteSessionStore
        store = SQLiteSessionStore(tmp_db)
        assert store.get("nonexistent-session-xyz") is None

    def test_session_delete_works(self, tmp_db):
        from backend.risk.session_store import SQLiteSessionStore, SessionState
        store = SQLiteSessionStore(tmp_db)
        state = SessionState(session_id="to-del", created_at=time.time(), last_updated_at=time.time())
        store.set("to-del", state, ttl_seconds=3600)
        assert store.get("to-del") is not None
        store.delete("to-del")
        assert store.get("to-del") is None


# ---------------------------------------------------------------------------
# 4. TTLCache
# ---------------------------------------------------------------------------
class TestTTLCache:

    def test_cached_value_returned_within_ttl(self):
        from backend.main import TTLCache
        cache = TTLCache(ttl_seconds=10)
        cache.set("key", {"value": 42})
        result = cache.get("key")
        assert result == {"value": 42}

    def test_returns_none_after_ttl_expiry(self):
        from backend.main import TTLCache
        cache = TTLCache(ttl_seconds=0)  # expires immediately
        cache.set("key", "data")
        time.sleep(0.01)  # ensure expiry
        result = cache.get("key")
        assert result is None

    def test_invalidate_clears_entry(self):
        from backend.main import TTLCache
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", "v")
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_returns_none_for_missing_key(self):
        from backend.main import TTLCache
        cache = TTLCache(ttl_seconds=10)
        assert cache.get("no-such-key") is None


# ---------------------------------------------------------------------------
# 5. AsyncTaskQueue
# ---------------------------------------------------------------------------
class TestAsyncTaskQueue:

    def test_enqueued_coroutine_is_executed(self):
        from backend.main import AsyncTaskQueue

        results = []

        async def worker():
            queue = AsyncTaskQueue(maxsize=10, num_workers=2)
            await queue.start()

            async def task(val):
                results.append(val)

            queue.enqueue(task, "hello")
            queue.enqueue(task, "world")
            await asyncio.sleep(0.1)  # let workers drain
            await queue.stop()

        asyncio.run(worker())
        assert sorted(results) == ["hello", "world"]

    def test_full_queue_does_not_block(self):
        from backend.main import AsyncTaskQueue
        import logging

        async def worker():
            # Very small queue — size 2
            queue = AsyncTaskQueue(maxsize=2, num_workers=0)  # no workers = won't drain

            async def noop():
                pass

            queue.enqueue(noop)
            queue.enqueue(noop)
            # Third enqueue should back-pressure gracefully (not raise)
            result = queue.enqueue(noop)
            assert result is False, "Should return False when queue is full"

        asyncio.run(worker())


# ---------------------------------------------------------------------------
# 6. Config fields
# ---------------------------------------------------------------------------
class TestScalabilityConfig:

    def test_default_session_ttl_hours(self):
        from backend.shared.config import Settings
        s = Settings()
        assert s.session_ttl_hours == 24

    def test_default_async_queue_size(self):
        from backend.shared.config import Settings
        s = Settings()
        assert s.async_queue_size == 500

    def test_default_metrics_cache_ttl(self):
        from backend.shared.config import Settings
        s = Settings()
        assert s.metrics_cache_ttl_s == 60

    def test_env_override_async_queue_size(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"CONTROLPLANE_ASYNC_QUEUE_SIZE": "200"}):
            from backend.shared.config import Settings
            s = Settings()
            assert s.async_queue_size == 200
