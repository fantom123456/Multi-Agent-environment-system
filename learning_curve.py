"""
Learning-curve view: lines up every run's task outcomes in chronological
order so you can see at a glance whether performance is trending up, flat,
or noisy across runs. This is the most direct way to answer "are my agents
actually learning?" -- if the shared policy (policy.db) is genuinely
improving, completion_rate and score should trend upward across successive
runs that reuse the same policy.db.

Usage: python learning_curve.py [--db runs.db] [--last N]
"""

import argparse
import sqlite3

# Same weights as score_tasks.py, so the score column here is directly
# comparable to that script's output.
R_COMPLETE = 10.0
R_PROGRESS = 0.2
C_SEND = 0.02
C_BCAST_EXTRA = 0.08
P_FAIL = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runs.db")
    ap.add_argument("--last", type=int, default=None, help="Only show the N most recent runs (default: all).")
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

    if not runs:
        raise SystemExit("No runs found in DB.")

    if args.last and len(runs) > args.last:
        runs = runs[-args.last:]

    header = f"{'#':>3} {'run_id':<12} {'completed':>9} {'failed':>6} {'completion_%':>12} {'contrib':>7} {'sent':>6} {'score':>8}"
    print(header)
    print("-" * len(header))

    completion_rates = []
    scores = []

    for idx, (run_id, _start_ms) in enumerate(runs, start=1):
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

        total = completed + failed
        completion_pct = (100.0 * completed / total) if total else float("nan")

        score = (
            R_COMPLETE * completed
            - P_FAIL * failed
            + R_PROGRESS * contrib
            - C_SEND * sent
            - C_BCAST_EXTRA * bcast
        )

        completion_rates.append(completion_pct)
        scores.append(score)

        short_id = run_id[:10] if len(run_id) > 10 else run_id
        print(f"{idx:>3} {short_id:<12} {completed:>9} {failed:>6} {completion_pct:>11.1f}% {contrib:>7} {sent:>6} {score:>8.2f}")

    con.close()

    # A rough, honest signal only -- not a statistical test. Compares the
    # average of the first half of runs to the second half.
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / mid
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
        print()
        print(f"Avg score, first half of runs:  {first_half_avg:.2f}")
        print(f"Avg score, second half of runs: {second_half_avg:.2f}")
        if second_half_avg > first_half_avg:
            print("-> Trending up: consistent with learning happening.")
        elif second_half_avg < first_half_avg:
            print("-> Trending down or flat: not clear evidence of learning yet.")
        else:
            print("-> No clear trend.")
    else:
        print("\n(Run at least 4 simulations with the same --policy-db to get a first-half vs second-half comparison.)")


if __name__ == "__main__":
    main()