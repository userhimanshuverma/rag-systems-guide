"""
SQLite-backed query cache.
Caches LLM responses keyed by a hash of (query + context fingerprint).
Avoids redundant LLM calls for identical queries.
"""

import sqlite3
import hashlib
import json
import time
import os
from app.config import CACHE_PATH
from app.logger import get_logger

logger = get_logger("cache")

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key      TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            created  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def make_key(query: str, context: str) -> str:
    """SHA-256 hash of query + context — unique cache key."""
    raw = f"{query.strip().lower()}||{context.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get(key: str) -> str | None:
    """Return cached response or None if not found."""
    with _get_conn() as conn:
        row = conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    if row:
        logger.info(f"Cache HIT for key={key[:12]}...")
        return row[0]
    return None


def set(key: str, response: str) -> None:
    """Store a response in the cache."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, response, created) VALUES (?, ?, ?)",
            (key, response, time.time())
        )
    logger.info(f"Cache SET for key={key[:12]}...")


def clear() -> int:
    """Clear all cached entries. Returns number of rows deleted."""
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        conn.execute("DELETE FROM cache")
    logger.info(f"Cache cleared — {count} entries removed")
    return count


def stats() -> dict:
    """Return cache statistics."""
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        oldest = conn.execute("SELECT MIN(created) FROM cache").fetchone()[0]
    return {"total_entries": count, "oldest_entry_ts": oldest}
