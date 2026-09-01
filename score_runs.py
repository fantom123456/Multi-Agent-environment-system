"""
Message-based scoring (v1): reward received messages, penalize sent
messages (broadcasts extra).

NOTE: superseded by score_runs_v2.py, which adds --db/--last args and
per-agent/per-second normalization. Kept for reference / diffing against
the newer version; prefer score_runs_v2.py for new analysis.
"""

import sqlite3
import json
from collections import defaultdict

RUNS_DB = "runs.db"


def main():
    con = sqlite3.connect(RUNS_DB)
    cur = con.cursor()

    run_ids = [r[0] for r in cur.execute(
        "SELECT DISTINCT run_id FROM events WHERE event_type='run_started' ORDER BY ts_ms"
    ).fetchall()]

    scores = defaultdict(float)

    for run_id, c in cur.execute("""
        SELECT run_id, COUNT(*)
        FROM events
        WHERE event_type='message_received'
        GROUP BY run_id
    """):
        scores[run_id] += 0.05 * c

    for run_id, payload_json in cur.execute("""
        SELECT run_id, payload_json
        FROM events
        WHERE event_type='message_sent'
    """):
        scores[run_id] -= 0.02
        if payload_json:
            kind = json.loads(payload_json).get("kind")
            if kind == "broadcast":
                scores[run_id] -= 0.03

    print("Run scores (higher is better under current reward):")
    for run_id in run_ids:
        print(run_id, round(scores[run_id], 2))

    con.close()


if __name__ == "__main__":
    main()