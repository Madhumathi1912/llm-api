import sqlite3
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "usage.db"  # same DB file as cost_logger


class TraceRepository:
    """
    Persisting trace steps. Nothing about
    timing or step orchestration — just storage, same spirit as
    UsageRepository in cost_logger.py.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._init_db()


    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()


    def insert_step(self, trace_id: str, step_name: str, duration_ms: float, metadata: dict) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                INSERT INTO trace_steps (trace_id, step_name, duration_ms, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    step_name,
                    duration_ms,
                    json.dumps(metadata),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            # Tracing should NEVER break the actual request — log and move on.
            logger.warning(f"Failed to persist trace step: {e}")


    def get_trace(self, trace_id: str) -> list[dict]:
        """Returns all steps for a given trace_id, in the order they were recorded."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT step_name, duration_ms, metadata, timestamp FROM trace_steps "
            "WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        conn.close()

        return [
            {
                "step_name": row[0],
                "duration_ms": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "timestamp": row[3],
            }
            for row in rows
        ]


class Tracer:
    """
    Timing named steps within a request and
    recording them via TraceRepository. Callers wrap each meaningful
    step (embed, search, generate, etc.) in a `with tracer.step(...)`
    block — duration is measured automatically.
    """

    def __init__(self, repository: TraceRepository):
        self._repository = repository

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    @contextmanager
    def step(self, trace_id: str, step_name: str, **metadata):
        """
        Usage:
            with tracer.step(trace_id, "vector_search", top_k=3):
                results = vector_store.search(...)

        Automatically measures how long the `with` block took and
        records it, regardless of whether the block succeeds or raises.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._repository.insert_step(trace_id, step_name, duration_ms, metadata)

    def get_trace(self, trace_id: str) -> list[dict]:
        return self._repository.get_trace(trace_id)


tracer = Tracer(repository=TraceRepository())