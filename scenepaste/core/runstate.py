"""Crash-safe resume state for large generation runs (SQLite, stdlib only)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Mapping, Optional

SCHEMA_VERSION = 1


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def stable_config_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class RunStateStore:
    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.state_dir = self.output_dir / ".scenepaste" / "runs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / f"{self.run_id}.sqlite3"
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                idx INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                stem TEXT,
                objects INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def set_meta(self, key: str, value) -> None:
        text = json.dumps(value, ensure_ascii=False, default=_json_default)
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, text))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return default

    def initialize(self, *, config_hash: str, count: int, config_payload: Mapping[str, object]) -> None:
        existing = self.get_meta("config_hash")
        if existing is not None and existing != config_hash:
            raise ValueError(
                f"Run {self.run_id} 的配置与恢复配置不一致。"
                "请使用原配置恢复，或指定新的 --run-id。"
            )
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.set_meta("run_id", self.run_id)
        self.set_meta("config_hash", config_hash)
        self.set_meta("count", int(count))
        self.set_meta("config", dict(config_payload))
        if self.get_meta("created_at") is None:
            self.set_meta("created_at", dt.datetime.now().isoformat(timespec="seconds"))
        self.set_meta("status", "running")

    def completed_bitmap(self, count: int) -> bytearray:
        bitmap = bytearray(max(0, int(count)))
        for (idx,) in self.conn.execute("SELECT idx FROM tasks WHERE status='completed' AND idx>=0 AND idx<?", (count,)):
            bitmap[int(idx)] = 1
        return bitmap

    def record_completed(self, idx: int, stem: str, objects: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks(idx,status,stem,objects,error,updated_at) VALUES(?,?,?,?,?,?)",
            (int(idx), "completed", str(stem), int(objects), None,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def record_failed(self, idx: int, error: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks(idx,status,stem,objects,error,updated_at) VALUES(?,?,?,?,?,?)",
            (int(idx), "failed", None, 0, str(error)[:2000],
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def counts(self) -> dict:
        rows = dict(self.conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall())
        objects = self.conn.execute("SELECT COALESCE(SUM(objects),0) FROM tasks WHERE status='completed'").fetchone()[0]
        return {"completed": int(rows.get("completed", 0)), "failed": int(rows.get("failed", 0)),
                "objects": int(objects or 0)}

    def mark_status(self, status: str) -> None:
        self.set_meta("status", status)
        self.set_meta("updated_at", dt.datetime.now().isoformat(timespec="seconds"))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def find_latest_resumable_run(output_dir: Path) -> Optional[str]:
    root = Path(output_dir) / ".scenepaste" / "runs"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            conn = sqlite3.connect(str(path))
            row = conn.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
            conn.close()
            status = json.loads(row[0]) if row else None
            if status != "completed":
                return path.stem
        except Exception:
            continue
    return None
