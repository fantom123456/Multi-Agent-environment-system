import argparse
import sqlite3
import json


def show_runs(db="runs.db"):
    con = sqlite3.connect(db)
    cur = con.cursor()
    print("tables:", cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    print("event count:", cur.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    print("top event types:")
    for et, c in cur.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY 2 DESC"):
        print(" ", et, c)

    print("\nmessage kinds sent:")
    for kind, c in cur.execute("""
        SELECT json_extract(payload_json,'$.kind') AS kind, COUNT(*)
        FROM events
        WHERE event_type='message_sent'
        GROUP BY kind
        ORDER BY 2 DESC
    """):
        print(" ", kind, c)
    con.close()


def show_policy(db="policy.db"):
    con = sqlite3.connect(db)
    cur = con.cursor()

    print("tables:", cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())

    print("arms:")
    for decision, arm, d in cur.execute("SELECT decision, arm, d FROM linucb_arms ORDER BY 1,2"):
        print(" ", decision, arm, d)

    print("\nb_json (learned reward-weighted feature sums):")
    for decision, arm, b_json in cur.execute("SELECT decision, arm, b_json FROM linucb_arms ORDER BY 1,2"):
        b = json.loads(b_json)
        print(" ", decision, arm, [round(x, 4) for x in b])

    print("\nA diagonal (should increase if updates happen):")
    for decision, arm, A_json in cur.execute("SELECT decision, arm, A_json FROM linucb_arms ORDER BY 1,2"):
        A = json.loads(A_json)
        diag = [round(A[i][i], 4) for i in range(len(A))]
        print(" ", decision, arm, diag)

    con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-db", default="runs.db")
    p.add_argument("--policy-db", default="policy.db")
    args = p.parse_args()

    show_runs(args.runs_db)
    print("\n" + "-" * 60 + "\n")
    show_policy(args.policy_db)


if __name__ == "__main__":
    main()