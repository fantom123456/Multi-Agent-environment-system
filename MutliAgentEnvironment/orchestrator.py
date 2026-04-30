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
    roles = []
    for i in range(n):
        roles.append(base[i % len(base)] + f"_{i//len(base)+1}")
    return roles


async def run_simulation(n_agents: int, duration_s: float, seed: int, db_path: str, policy_db: str):
    run_id = str(uuid.uuid4())

    bus = InMemoryBus()
    telemetry = Telemetry(db_path=db_path)
    await telemetry.start()

    # ONE global shared policy
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

    # Build role -> agents mapping for targeted notifications
    role_to_agents: dict[str, list[str]] = {}
    for aid, role in zip(agent_ids, roles):
        role_to_agents.setdefault(role, []).append(aid)

    # Task system
    tm = TaskManager(
        rng=rng,
        role_pool=list(set(roles)),
        episode_duration_s=duration_s,
        expected_tasks=6,
    )
    tm.start()

    agents: list[Agent] = []
    tasks: list[asyncio.Task] = []

    proactive_interval_s = rng.uniform(0.7, 1.5)  # shared cadence (policy picks action)

    for i, aid in enumerate(agent_ids):
        inbox = bus.register_agent(aid)
        cfg = AgentConfig(
            agent_id=aid,
            role=roles[i],
            seed=rng.randint(1, 1_000_000_000),
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
            policy=policy,
            proactive_interval_s=proactive_interval_s,
            task_manager=tm,
        )
        agents.append(agent)

    for agent in agents:
        tasks.append(asyncio.create_task(agent.run()))

    # escalation tracking so we don't spam multiple times per task
    task_escalation_state: dict[str, dict[str, bool]] = {}

    # Orchestrator task loop
    end_ts = asyncio.get_event_loop().time() + duration_s
    while asyncio.get_event_loop().time() < end_ts:
        # spawn new tasks
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

            # Create escalation state
            task_escalation_state[t.task_id] = {"extra_sent": False, "broadcast_sent": False}

            # Step 0: targeted notify (required roles only)
            targets: list[str] = []
            for role in t.required_roles:
                targets.extend(role_to_agents.get(role, []))

            # Fallback: if no targets, notify a small subset
            if not targets:
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

        # Escalation logic for open tasks (anti-noise: only broadcast if needed)
        for t in tm.open_tasks():
            st = task_escalation_state.get(t.task_id)
            if not st:
                continue

            time_left = t.deadline_ts - time.time()

            # Escalation 1: notify a few extra agents when time is getting short
            if (not st["extra_sent"]) and time_left <= 8.0:
                st["extra_sent"] = True
                extra = rng.sample(agent_ids, k=min(8, len(agent_ids)))
                for to_aid in extra:
                    msg = bus.new_message(
                        from_agent_id="system",
                        to_agent_id=to_aid,
                        kind="task_request",
                        body={
                            "task_id": t.task_id,
                            "deadline_ts": t.deadline_ts,
                            "required_roles": t.required_roles,
                            "required_contributions": t.required_contributions,
                            "escalation": "extra",
                        },
                    )
                    await bus.send(msg)

                await telemetry.log(
                    new_event(
                        run_id=run_id,
                        event_type="task_escalated",
                        payload={"task_id": t.task_id, "level": "extra", "time_left_s": time_left},
                    )
                )

            # Escalation 2: last resort broadcast very near deadline
            if (not st["broadcast_sent"]) and time_left <= 3.0:
                st["broadcast_sent"] = True
                msg = bus.new_message(
                    from_agent_id="system",
                    to_agent_id=None,
                    kind="task_request",
                    body={
                        "task_id": t.task_id,
                        "deadline_ts": t.deadline_ts,
                        "required_roles": t.required_roles,
                        "required_contributions": t.required_contributions,
                        "escalation": "broadcast",
                    },
                )
                await bus.send(msg)

                await telemetry.log(
                    new_event(
                        run_id=run_id,
                        event_type="task_escalated",
                        payload={"task_id": t.task_id, "level": "broadcast", "time_left_s": time_left},
                    )
                )

        # fail overdue tasks
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

    # persist learned policy
    policy.save()

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