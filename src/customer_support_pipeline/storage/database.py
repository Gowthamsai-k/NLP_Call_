"""
SQLite Storage Backend for Customer Support Call Evaluations.
Provides ACID transactions, indexed queries, zero external dependencies,
and constant O(1) memory retrieval for 1,000 to 1,000,000+ call records.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import sqlite3
    sqlite3.connect(":memory:").close()
except Exception:
    try:
        import pysqlite3 as sqlite3
    except Exception as e:
        raise ImportError(f"Neither standard sqlite3 nor pysqlite3 could be loaded: {e}")

logger = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = "evaluations.db"


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Creates a SQLite connection with WAL mode and fast synchronous settings."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64 MB cache
    return conn


def init_db(db_path: Path) -> None:
    """Initializes SQLite schema and indexes for high-throughput call evaluation queries."""
    with get_db_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                agent TEXT,
                department TEXT,
                queue_id TEXT,
                customer_phone TEXT,
                call_date TEXT,
                dominant_language TEXT,
                intro_compliant INTEGER DEFAULT 0,
                intro_status TEXT,
                total_score REAL,
                professionalism_score REAL,
                respectfulness_score REAL,
                politeness_score REAL,
                resolution_status TEXT,
                duration_seconds REAL,
                diarization_timeline TEXT,
                data_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_agent ON evaluations(agent);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_resolution ON evaluations(resolution_status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_dominant_lang ON evaluations(dominant_language);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_intro ON evaluations(intro_compliant);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluations(created_at);")
        conn.commit()


def parse_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extracts Next Education metadata from standardized audio filename."""
    if not filename:
        return {"agent": "Support Executive", "dept": "Support", "queue": "General", "phone": "Confidential", "date": ""}
    stem = filename.replace(".mp3", "").replace(".wav", "").replace(".m4a", "").replace(".json", "")
    parts = stem.split("__")
    if len(parts) >= 6:
        agent_raw = parts[0]
        if "_nexteducation_in" in agent_raw:
            agent = agent_raw.replace("_nexteducation_in", "@nexteducation.in").replace("_", ".").replace(".@", "@")
        else:
            agent = agent_raw.replace("_", " ")
        return {
            "agent": agent,
            "dept": parts[1] or "Support",
            "queue": parts[2] or "General",
            "phone": parts[4] or "Confidential",
            "date": (parts[5] or "").replace("_", " ")
        }
    return {"agent": "Support Executive", "dept": "Support", "queue": "General", "phone": "Confidential", "date": ""}


def save_evaluation_to_db(db_path: Path, eval_id: str, filename: str, data: Dict[str, Any]) -> None:
    """Saves or updates a single call evaluation record atomically into SQLite."""
    init_db(db_path)
    meta = parse_metadata_from_filename(filename or data.get("audio_file", ""))
    
    sc = data.get("scorecard", {}) or {}
    total_score = float(sc.get("total_score")) if sc.get("total_score") is not None else None
    prof_score = float(sc.get("professionalism_score")) if sc.get("professionalism_score") is not None else None
    resp_score = float(sc.get("respectfulness_score")) if sc.get("respectfulness_score") is not None else None
    pol_score = float(sc.get("politeness_score")) if sc.get("politeness_score") is not None else None

    res = data.get("resolution", {}) or {}
    res_status = res.get("status", "unclear")

    lang_data = data.get("language_analysis", {}) or {}
    dom_lang = lang_data.get("dominant_language", "English")

    intro_data = data.get("introduction_check", {}) or {}
    intro_comp = 1 if intro_data.get("compliant") else 0
    intro_status = intro_data.get("status", "non_compliant" if not intro_comp else "compliant")

    dt = data.get("diarization_timeline", {}) or {}
    dur_sec = float(dt.get("total_duration_seconds", 0.0))
    dt_json = json.dumps(dt, ensure_ascii=False) if dt else None

    data_json = json.dumps(data, ensure_ascii=False)

    with get_db_connection(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO evaluations (
                id, filename, agent, department, queue_id, customer_phone, call_date,
                dominant_language, intro_compliant, intro_status,
                total_score, professionalism_score, respectfulness_score, politeness_score,
                resolution_status, duration_seconds, diarization_timeline, data_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            );
        """, (
            eval_id, filename, meta["agent"], meta["dept"], meta["queue"], meta["phone"], meta["date"],
            dom_lang, intro_comp, intro_status,
            total_score, prof_score, resp_score, pol_score,
            res_status, dur_sec, dt_json, data_json
        ))
        conn.commit()


def query_evaluations_from_db(
    db_path: Path,
    limit: int = 1000,
    offset: int = 0,
    agent: Optional[str] = None,
    resolution: Optional[str] = None,
    dominant_language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Queries call evaluation records with pagination and indexed filters."""
    if not db_path.exists():
        return []

    conditions = []
    params: List[Any] = []

    if agent and agent.lower() != "all":
        conditions.append("agent = ?")
        params.append(agent)

    if resolution and resolution.lower() != "all":
        conditions.append("resolution_status = ?")
        params.append(resolution)

    if dominant_language and dominant_language.lower() != "all":
        conditions.append("dominant_language = ?")
        params.append(dominant_language)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT id, filename, data_json
        FROM evaluations
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?;
    """
    params.extend([limit, offset])

    results = []
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(query, params)
        for row in cursor.fetchall():
            try:
                data = json.loads(row["data_json"])
                results.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "data": data,
                })
            except Exception:
                continue

    return results


def sync_disk_evaluations_to_db(output_dir: Path, db_path: Path) -> int:
    """Scans output directory and syncs all existing JSON evaluation files into SQLite."""
    if not output_dir.exists():
        return 0

    init_db(db_path)
    synced_count = 0

    for eval_path in output_dir.rglob("call_evaluation.json"):
        if "failed" in eval_path.parts:
            continue
        try:
            eval_id = str(eval_path.parent.name)
            with eval_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            call_name = data.get("audio_file") or eval_id
            save_evaluation_to_db(db_path, eval_id, call_name, data)
            synced_count += 1
        except Exception as e:
            logger.debug(f"Sync error for {eval_path}: {e}")

    logger.info(f"SQLite Sync: Indexed {synced_count} evaluation record(s) into {db_path.name}")
    return synced_count
