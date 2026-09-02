import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiosqlite


def utc_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Event:
    event_id: str
    ts_ms: int
    run_id: str
    event_type: str

    agent_id: Optional[str] = None
    from_agent_id: Optional[str] = None
    to_agent_id: Optional[str] = None

    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

    payload: Optional[Dict[str, Any]] = None


class Telemetry:
    """
    SQLite telemetry logger.
    Stores one row per event with payload as JSON (payload_json).
    """

    def __init__(self, db_path: str = "runs.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def start(self):
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                ts_ms INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,

                agent_id TEXT,
                from_agent_id TEXT,
                to_agent_id TEXT,

                conversation_id TEXT,
                message_id TEXT,

                payload_json TEXT
            );
            """
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts_ms);"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);"
        )
        await self._conn.commit()

    async def stop(self):
        if self._conn is not None:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None

    async def log(self, event: Event):
        if self._conn is None:
            raise RuntimeError("Telemetry not started. Call await telemetry.start().")

        payload_json = (
            json.dumps(event.payload, separators=(",", ":"), ensure_ascii=False)
            if event.payload
            else None
        )
        await self._conn.execute(
            """
            INSERT INTO events (
                event_id, ts_ms, run_id, event_type,
                agent_id, from_agent_id, to_agent_id,
                conversation_id, message_id,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event.event_id,
                event.ts_ms,
                event.run_id,
                event.event_type,
                event.agent_id,
                event.from_agent_id,
                event.to_agent_id,
                event.conversation_id,
                event.message_id,
                payload_json,
            ),
        )

    async def flush(self):
        if self._conn is not None:
            await self._conn.commit()


def new_event(
    run_id: str,
    event_type: str,
    agent_id: Optional[str] = None,
    from_agent_id: Optional[str] = None,
    to_agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Event:
    return Event(
        event_id=str(uuid.uuid4()),
        ts_ms=utc_ms(),
        run_id=run_id,
        event_type=event_type,
        agent_id=agent_id,
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        conversation_id=conversation_id,
        message_id=message_id,
        payload=payload,
    )