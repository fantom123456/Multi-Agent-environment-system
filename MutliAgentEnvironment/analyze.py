import argparse
import json
import sqlite3
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runs.db")
    parser.add_argument("--run-id", default=None, help="If not provided, uses the most recent run_id.")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    run_id = args.run_id
    if run_id is None:
        row = cur.execute(
            "SELECT run_id FROM events WHERE event_type='run_started' ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        if not row:
            raise SystemExit("No runs found in DB.")
        run_id = row[0]

    print(f"Using run_id: {run_id}")

    # Event counts
    rows = cur.execute(
        "SELECT event_type, COUNT(*) FROM events WHERE run_id=? GROUP BY event_type ORDER BY 2 DESC",
        (run_id,),
    ).fetchall()
    print("\nEvent counts:")
    for et, c in rows:
        print(f"- {et}: {c}")

    # Top talkers (sent)
    rows = cur.execute(
        """
        SELECT from_agent_id, COUNT(*)
        FROM events
        WHERE run_id=? AND event_type='message_sent'
        GROUP BY from_agent_id
        ORDER BY 2 DESC
        LIMIT 10
        """,
        (run_id,),
    ).fetchall()
    print("\nTop 10 senders:")
    for aid, c in rows:
        print(f"- {aid}: {c}")

    # Who talks to whom (direct only)
    rows = cur.execute(
        """
        SELECT from_agent_id, to_agent_id, COUNT(*)
        FROM events
        WHERE run_id=? AND event_type='message_sent' AND to_agent_id IS NOT NULL
        GROUP BY from_agent_id, to_agent_id
        ORDER BY 3 DESC
        LIMIT 15
        """,
        (run_id,),
    ).fetchall()
    print("\nTop 15 directed edges (from -> to):")
    for f, t, c in rows:
        print(f"- {f} -> {t}: {c}")

    rows = cur.execute(
        "SELECT event_type, COUNT(*) FROM events WHERE run_id=? AND event_type LIKE 'task_%' GROUP BY event_type",
        (run_id,),
    ).fetchall()
    print("\nTask event counts:")
    for et, c in rows:
        print(f"- {et}: {c}")

    con.close()


if __name__ == "__main__":
    main()