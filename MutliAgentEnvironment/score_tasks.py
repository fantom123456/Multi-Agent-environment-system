import argparse
import sqlite3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runs.db")
    ap.add_argument("--last", type=int, default=5)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    runs = cur.execute("""
        SELECT run_id, MIN(ts_ms) AS start_ms
        FROM events
        WHERE event_type='run_started'
        GROUP BY run_id
        ORDER BY start_ms
    """).fetchall()

    if args.last and len(runs) > args.last:
        runs = runs[-args.last:]

    # weights (match Agent)
    R_COMPLETE = 10.0
    R_PROGRESS = 0.2
    C_SEND = 0.02
    C_BCAST_EXTRA = 0.08
    P_FAIL = 5.0

    print(f"Task-based score (last {args.last} runs):")
    for run_id, _ in runs:
        completed = cur.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_type='task_completed'", (run_id,)
        ).fetchone()[0]
        failed = cur.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_type='task_failed'", (run_id,)
        ).fetchone()[0]
        contrib = cur.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_type='task_contributed'", (run_id,)
        ).fetchone()[0]

        sent = cur.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_type='message_sent'", (run_id,)
        ).fetchone()[0]
        bcast = cur.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_type='message_sent' AND json_extract(payload_json,'$.kind')='broadcast'",
            (run_id,),
        ).fetchone()[0]

        score = (
            R_COMPLETE * completed
            - P_FAIL * failed
            + R_PROGRESS * contrib
            - C_SEND * sent
            - C_BCAST_EXTRA * bcast
        )

        print(f"\nrun_id: {run_id}")
        print(f"  tasks_completed: {completed}")
        print(f"  tasks_failed:    {failed}")
        print(f"  contributions:   {contrib}")
        print(f"  sent:            {sent} (broadcast {bcast})")
        print(f"  score:           {score:.2f}")

    con.close()

if __name__ == "__main__":
    main()