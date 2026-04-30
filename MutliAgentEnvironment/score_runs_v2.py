import argparse
import sqlite3
import json

RUNS_DB = "runs.db"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=RUNS_DB)
    ap.add_argument("--last", type=int, default=5)   # <--- add this
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    starts = cur.execute("""
        SELECT run_id,
               MIN(ts_ms) AS start_ms,
               MAX(CASE WHEN event_type='run_started' THEN json_extract(payload_json,'$.n_agents') END) AS n_agents,
               MAX(CASE WHEN event_type='run_started' THEN json_extract(payload_json,'$.duration_s') END) AS duration_s
        FROM events
        WHERE event_type IN ('run_started')
        GROUP BY run_id
        ORDER BY start_ms
    """).fetchall()

    # keep only last N
    if args.last and len(starts) > args.last:
        starts = starts[-args.last:]

    stats = {run_id: {"sent": 0, "recv": 0, "broadcast": 0} for (run_id, *_rest) in starts}

    for run_id, c in cur.execute("""
        SELECT run_id, COUNT(*)
        FROM events
        WHERE event_type='message_received'
        GROUP BY run_id
    """):
        if run_id in stats:
            stats[run_id]["recv"] = c

    for run_id, payload_json in cur.execute("""
        SELECT run_id, payload_json
        FROM events
        WHERE event_type='message_sent'
    """):
        if run_id not in stats:
            continue
        stats[run_id]["sent"] += 1
        try:
            kind = json.loads(payload_json).get("kind")
        except Exception:
            kind = None
        if kind == "broadcast":
            stats[run_id]["broadcast"] += 1

    RECV_R = 0.05
    SEND_COST = 0.02
    BCAST_EXTRA = 0.03

    print("Per-run metrics (last", args.last, "runs):")
    for run_id, start_ms, n_agents, duration_s in starts:
        s = stats[run_id]["sent"]
        r = stats[run_id]["recv"]
        b = stats[run_id]["broadcast"]
        score = RECV_R * r - SEND_COST * s - BCAST_EXTRA * b

        duration_s = float(duration_s) if duration_s else None
        n_agents = float(n_agents) if n_agents else None
        score_per_agent_s = (score / (duration_s * n_agents)) if (duration_s and n_agents) else None

        print(f"\nrun_id: {run_id}")
        print(f"  agents: {n_agents}  duration_s: {duration_s}")
        print(f"  sent: {s}  recv: {r}  broadcast_sent: {b}")
        print(f"  score_total: {score:.2f}")
        if score_per_agent_s is not None:
            print(f"  score_per_agent_s: {score_per_agent_s:.6f}")

    con.close()

if __name__ == "__main__":
    main()