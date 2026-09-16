import argparse
import asyncio
import random
import uuid
import time

from agent import Agent, AgentConfig
from bus import InMemoryBus
from policy import GlobalPolicy
from telemetry import Telemetry, new_event
from tasks import TaskManager


def make_roles(n: int) -> list[str]:
    """
    Assign each agent one of a fixed set of role names, cycling through the
    pool so multiple agents genuinely share each role (e.g. with n=50 and
    10 base roles, ~5 agents get "planner", ~5 get "critic", etc).

    IMPORTANT: earlier versions of this function appended a unique suffix
    per cycle (e.g. "planner_1", "planner_2", ...), which meant every agent
    ended up with an effectively unique role string -- fine when
    `require_role_match=False` (role was only a reward bonus), but a hard
    bug once `require_role_match=True` actually gates who may contribute:
    a task requiring a role that only 1 agent holds can never reach a
    multi-agent contribution quota. Do not reintroduce a per-agent-unique
    suffix here unless required_contributions is also capped to match the
    resulting (tiny) eligible-agent pool size.
    """
    base = [
        "planner",
        "critic",
        "builder",
        "negotiator",
        "researcher",
        "summarizer",
        "moderator",
        "explorer",
        "optimizer",
        "skeptic",
    ]
    return [base[i % len(base)] for i in range(n)]


async def run_simulation(n_agents: int, duration_s: float, seed: int, db_path: str, policy_db: str):
    run_id = str(uuid.uuid4())

    bus = InMemoryBus()
    telemetry = Telemetry(db_path=db_path)
    await telemetry.start()

    # One policy shared by every agent (see policy.py docstring).
    policy = GlobalPolicy(db_path=policy_db)

    await telemetry.log(
        new_event(
            run_id=run_id,
            event_type="run_started",
            payload={
                "n_agents": n_agents,
                "duration_s": duration_s,
                "seed": seed,
                "policy_db": policy_db,
                "tasks_expected": 6,
                "task_notify_mode": "escalation_v1",
            },
        )
    )

    rng = random.Random(seed)
    agent_ids = [f"agent_{i:03d}" for i in range(n_agents)]
    roles = make_roles(n_agents)

    # Role -> agent_ids, so task notifications can target agents whose role matches.
    role_to_agents: dict[str, list[str]] = {}
    for aid, role in zip(agent_ids, roles):
        role_to_agents.setdefault(role, []).append(aid)

    # Inverse lookup (agent_id -> role) so agents can tell which of their
    # known peers match a given task's required roles when picking who to
    # invite via "collab" -- see Agent._pick_peer.
    agent_id_to_role: dict[str, str] = dict(zip(agent_ids, roles))

    tm = TaskManager(
        rng=rng,
        role_pool=list(set(roles)),
        episode_duration_s=duration_s,
        expected_tasks=6,
        # --- tuned for difficulty: forces real scarcity instead of "any of
        # 50 agents can fill a 4-10 quota", but still winnable -- an earlier
        # pass at these numbers (8-18s deadlines, 8-18 quota) produced 0%
        # completion across 10 runs even with a converged policy, meaning
        # it was structurally impossible, not just hard. Loosened below.
        min_deadline_s=14.0,       # was 10.0 originally / 8.0 in the too-hard attempt
        max_deadline_s=25.0,       # was 30.0 originally / 18.0 in the too-hard attempt
        min_required_roles=2,
        max_required_roles=2,      # was up to 3 originally -- narrower eligible pool
        min_contrib=5,             # was 4 originally / 8 in the too-hard attempt
        max_contrib=10,            # was 10 originally / 18 in the too-hard attempt
        require_role_match=True,   # was False originally -- ONLY matching-role agents can contribute
    )
    tm.start()

    agents: list[Agent] = []
    tasks: list[asyncio.Task] = []

    for i, aid in enumerate(agent_ids):
        inbox = bus.register_agent(aid)
        cfg = AgentConfig(
            agent_id=aid,
            role=roles[i],
            seed=rng.randint(1, 1_000_000_000),
            # RESERVED: these four fields are only consumed by a future
            # proactive-mode Agent (see agent.py). The current Agent never
            # reads them, so they have no effect on simulation behavior today.
            proactive_interval_s=rng.uniform(0.7, 2.0),
            broadcast_probability=rng.uniform(0.10, 0.35),
            reply_probability=rng.uniform(0.50, 0.90),
            collaboration_probability=rng.uniform(0.05, 0.30),
        )
        agent = Agent(
            run_id=run_id,
            cfg=cfg,
            bus=bus,
            inbox=inbox,
            telemetry=telemetry,
            all_agent_ids=agent_ids,
            all_agent_roles=agent_id_to_role,
            policy=policy,
            task_manager=tm,
        )
        agents.append(agent)

    for agent in agents:
        tasks.append(asyncio.create_task(agent.run()))

    # Tracks which escalation levels have already fired per task, so we
    # don't re-send the same escalation notification repeatedly.
    task_escalation_state: dict[str, dict[str, bool]] = {}

    end_ts = asyncio.get_event_loop().time() + duration_s
    while asyncio.get_event_loop().time() < end_ts:
        # --- spawn newly-due tasks and notify the agents whose role matches ---
        new_tasks = tm.tick_spawn()
        for t in new_tasks:
            await telemetry.log(
                new_event(
                    run_id=run_id,
                    event_type="task_spawned",
                    payload={
                        "task_id": t.task_id,
                        "deadline_ts": t.deadline_ts,
                        "required_roles": t.required_roles,
                        "required_contributions": t.required_contributions,
                    },
                )
            )

            task_escalation_state[t.task_id] = {"extra_sent": False, "broadcast_sent": False}

            # Step 0: targeted notify — only agents with a matching role.
            # Only tell a SAMPLE of each required role directly -- not every
            # eligible agent. If every role-matching agent were notified up
            # front, reaching quota would only ever depend on reply-vs-ignore
            # (which the policy already solves trivially), making "collab"
            # structurally useless: there'd be no one left worth recruiting.
            # Under-covering here means the rest of the eligible pool can
            # only be reached via a directly-notified agent's "collab"
            # invite (see Agent._pick_peer, which prefers role-matching
            # known peers) or via escalation below.
            INITIAL_NOTIFY_PER_ROLE = 2
            targets: list[str] = []
            for role in t.required_roles:
                pool = role_to_agents.get(role, [])
                targets.extend(rng.sample(pool, k=min(INITIAL_NOTIFY_PER_ROLE, len(pool))))

            if not targets:
                # Fallback if no agent matches any required role.
                targets = rng.sample(agent_ids, k=min(5, len(agent_ids)))

            for to_aid in targets:
                msg = bus.new_message(
                    from_agent_id="system",
                    to_agent_id=to_aid,
                    kind="task_request",
                    body={
                        "task_id": t.task_id,
                        "deadline_ts": t.deadline_ts,
                        "required_roles": t.required_roles,
                        "required_contributions": t.required_contributions,
                    },
                )
                await bus.send(msg)

        # --- escalate notifications for open tasks running low on time ---
        # This keeps message volume down early (only role-matched agents get
        # pinged) while still giving tasks a chance to complete under time
        # pressure by widening the audience as the deadline nears.
        for t in tm.open_tasks():
            st = task_escalation_state.get(t.task_id)
            if not st:
                continue

            time_left = t.deadline_ts - time.time()

            # --- EXPERIMENT: DISABLED ESCALATIONS ---
            # Escalation 1: notify a wider (but still limited) set of agents.
            # if (not st["extra_sent"]) and time_left <= 8.0:
            #     st["extra_sent"] = True
            #     extra = rng.sample(agent_ids, k=min(8, len(agent_ids)))
            #     for to_aid in extra:
            #         msg = bus.new_message(
            #             from_agent_id="system",
            #             to_agent_id=to_aid,
            #             kind="task_request",
            #             body={
            #                 "task_id": t.task_id,
            #                 "deadline_ts": t.deadline_ts,
            #                 "required_roles": t.required_roles,
            #                 "required_contributions": t.required_contributions,
            #                 "escalation": "extra",
            #             },
            #         )
            #         await bus.send(msg)
            # 
            #     await telemetry.log(
            #         new_event(
            #             run_id=run_id,
            #             event_type="task_escalated",
            #             payload={"task_id": t.task_id, "level": "extra", "time_left_s": time_left},
            #         )
            #     )

            # Escalation 2: last resort — broadcast to everyone.
            # if (not st["broadcast_sent"]) and time_left <= 3.0:
            #     st["broadcast_sent"] = True
            #     msg = bus.new_message(
            #         from_agent_id="system",
            #         to_agent_id=None,
            #         kind="task_request",
            #         body={
            #             "task_id": t.task_id,
            #             "deadline_ts": t.deadline_ts,
            #             "required_roles": t.required_roles,
            #             "required_contributions": t.required_contributions,
            #             "escalation": "broadcast",
            #         },
            #     )
            #     await bus.send(msg)
            # 
            #     await telemetry.log(
            #         new_event(
            #             run_id=run_id,
            #             event_type="task_escalated",
            #             payload={"task_id": t.task_id, "level": "broadcast", "time_left_s": time_left},
            #         )
            #     )

        # --- fail any tasks that ran out the clock ---
        failed = tm.tick_deadlines()
        for t in failed:
            await telemetry.log(
                new_event(
                    run_id=run_id,
                    event_type="task_failed",
                    payload={"task_id": t.task_id, "failed_ts": t.failed_ts},
                )
            )

        await asyncio.sleep(0.25)

    for agent in agents:
        await agent.stop()

    await asyncio.gather(*tasks, return_exceptions=True)

    await telemetry.log(new_event(run_id=run_id, event_type="run_stopped"))
    await telemetry.flush()
    await telemetry.stop()

    policy.save()  # persist learned bandit weights across runs

    print(f"Run complete. run_id={run_id}")
    print(f"Telemetry database: {db_path}")
    print(f"Policy database: {policy_db}")
    return run_id


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=50, help="Number of agents (50-100 recommended).")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for the run.")
    parser.add_argument("--db", type=str, default="runs.db", help="SQLite DB path.")
    parser.add_argument("--policy-db", type=str, default="policy.db", help="SQLite policy DB path.")
    args = parser.parse_args()

    if args.agents < 1:
        raise SystemExit("--agents must be >= 1")

    await run_simulation(
        n_agents=args.agents,
        duration_s=args.duration,
        seed=args.seed,
        db_path=args.db,
        policy_db=args.policy_db,
    )


if __name__ == "__main__":
    asyncio.run(main())