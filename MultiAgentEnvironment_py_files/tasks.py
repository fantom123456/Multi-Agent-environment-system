import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Task:
    task_id: str
    spawn_ts: float
    deadline_ts: float
    required_roles: List[str]
    required_contributions: int
    contributions_received: int = 0
    contributor_ids: Set[str] = field(default_factory=set)
    contributor_contexts: List[tuple] = field(default_factory=list)
    completed: bool = False
    failed: bool = False
    completed_ts: Optional[float] = None
    failed_ts: Optional[float] = None


@dataclass
class InviteRecord:
    inviter_id: str
    inviter_ctx: tuple
    invited_id: str
    task_id: str


class TaskManager:
    def __init__(
        self,
        rng,
        role_pool: List[str],
        episode_duration_s: float,
        expected_tasks: int = 6,
        min_deadline_s: float = 14.0,
        max_deadline_s: float = 25.0,
        min_required_roles: int = 2,
        max_required_roles: int = 2,
        min_contrib: int = 5,
        max_contrib: int = 10,
        require_role_match: bool = True,
    ):
        self.rng = rng
        self.role_pool = role_pool
        self.episode_duration_s = episode_duration_s
        self.expected_tasks = expected_tasks
        self.min_deadline_s = min_deadline_s
        self.max_deadline_s = max_deadline_s
        self.min_required_roles = min_required_roles
        self.max_required_roles = max_required_roles
        self.min_contrib = min_contrib
        self.max_contrib = max_contrib
        self.require_role_match = require_role_match

        self.tasks: Dict[str, Task] = {}
        self.invites: List[InviteRecord] = []
        self._spawn_times: List[float] = []
        self._next_task_idx = 0
        self._start_ts: float = 0.0

    def start(self):
        self._start_ts = time.time()
        interval = self.episode_duration_s / (self.expected_tasks + 1)
        self._spawn_times = [
            (i + 1) * interval + self.rng.uniform(-interval * 0.25, interval * 0.25)
            for i in range(self.expected_tasks)
        ]
        self._spawn_times.sort()

    def tick_spawn(self) -> List[Task]:
        elapsed = time.time() - self._start_ts
        spawned = []
        while self._next_task_idx < len(self._spawn_times) and elapsed >= self._spawn_times[self._next_task_idx]:
            tid = f"task_{self._next_task_idx:03d}"
            d_len = self.rng.uniform(self.min_deadline_s, self.max_deadline_s)
            req_roles = self.rng.sample(self.role_pool, k=min(self.min_required_roles, len(self.role_pool)))
            req_contrib = self.rng.randint(self.min_contrib, self.max_contrib)

            t = Task(
                task_id=tid,
                spawn_ts=time.time(),
                deadline_ts=time.time() + d_len,
                required_roles=req_roles,
                required_contributions=req_contrib,
            )
            self.tasks[tid] = t
            spawned.append(t)
            self._next_task_idx += 1
        return spawned

    def tick_deadlines(self) -> List[Task]:
        now = time.time()
        failed = []
        for t in self.tasks.values():
            if not t.completed and not t.failed and now >= t.deadline_ts:
                t.failed = True
                t.failed_ts = now
                failed.append(t)
        return failed

    def record_invite(self, task_id: str, inviter_id: str, inviter_ctx: tuple, invited_id: str):
        self.invites.append(InviteRecord(inviter_id, inviter_ctx, invited_id, task_id))

    def contribute(self, task_id: str, agent_id: str, agent_role: str, ctx: tuple) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if not task or task.completed or task.failed:
            return {"ok": False, "reason": "inactive"}

        if agent_id in task.contributor_ids:
            return {"ok": False, "reason": "duplicate"}

        role_match = agent_role in task.required_roles if self.require_role_match else True
        if self.require_role_match and not role_match:
            return {"ok": False, "reason": "role_mismatch"}

        task.contributions_received += 1
        task.contributor_ids.add(agent_id)
        task.contributor_contexts.append((agent_id, ctx))

        invite_credit = None
        inviter_id = None
        for inv in list(self.invites):
            if inv.task_id == task_id and inv.invited_id == agent_id:
                invite_credit = inv.inviter_ctx
                inviter_id = inv.inviter_id
                self.invites.remove(inv)
                break

        newly_completed = False
        if task.contributions_received >= task.required_contributions:
            task.completed = True
            task.completed_ts = time.time()
            newly_completed = True

        return {
            "ok": True,
            "role_match": role_match,
            "remaining": max(0, task.required_contributions - task.contributions_received),
            "invite_credit": invite_credit,
            "inviter_id": inviter_id,
            "newly_completed": newly_completed,
            "required_contributions": task.required_contributions,
            "contributor_contexts": list(task.contributor_contexts),
        }