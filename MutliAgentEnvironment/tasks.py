import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Task:
    task_id: str
    created_ts: float
    deadline_ts: float
    required_roles: List[str]
    required_contributions: int

    contributions: int = 0
    contributors: Set[str] = field(default_factory=set)

    completed_ts: Optional[float] = None
    failed_ts: Optional[float] = None

    def is_open(self) -> bool:
        return self.completed_ts is None and self.failed_ts is None

    def is_completed(self) -> bool:
        return self.completed_ts is not None

    def is_failed(self) -> bool:
        return self.failed_ts is not None


class TaskManager:
    """
    Cooperative task system:
      - tasks arrive over time
      - agents contribute at most once per task (true teamwork)
      - tasks complete when enough unique contributors contribute before deadline
    """

    def __init__(
        self,
        *,
        rng: random.Random,
        role_pool: List[str],
        episode_duration_s: float,
        expected_tasks: int = 6,
        min_deadline_s: float = 10.0,
        max_deadline_s: float = 30.0,
        min_required_roles: int = 2,
        max_required_roles: int = 3,
        min_contrib: int = 4,
        max_contrib: int = 10,
        require_role_match: bool = False,  # set True if you want strict role gating
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

        # schedule task arrivals uniformly (simple)
        self.spawn_times = sorted(
            self.rng.uniform(0.0, max(0.1, episode_duration_s - 1.0))
            for _ in range(expected_tasks)
        )
        self._spawn_idx = 0
        self._t0: Optional[float] = None

    def start(self):
        self._t0 = time.time()

    def now(self) -> float:
        return time.time()

    def elapsed(self) -> float:
        assert self._t0 is not None
        return self.now() - self._t0

    def tick_spawn(self) -> List[Task]:
        if self._t0 is None:
            raise RuntimeError("TaskManager.start() not called")

        new_tasks: List[Task] = []
        while self._spawn_idx < len(self.spawn_times) and self.elapsed() >= self.spawn_times[self._spawn_idx]:
            t = self._spawn_one()
            new_tasks.append(t)
            self._spawn_idx += 1
        return new_tasks

    def _spawn_one(self) -> Task:
        task_id = str(uuid.uuid4())
        created = self.now()
        deadline = created + self.rng.uniform(self.min_deadline_s, self.max_deadline_s)

        k_roles = self.rng.randint(self.min_required_roles, self.max_required_roles)
        required_roles = self.rng.sample(self.role_pool, k=min(k_roles, len(self.role_pool)))

        required_contrib = self.rng.randint(self.min_contrib, self.max_contrib)

        task = Task(
            task_id=task_id,
            created_ts=created,
            deadline_ts=deadline,
            required_roles=required_roles,
            required_contributions=required_contrib,
        )
        self.tasks[task_id] = task
        return task

    def tick_deadlines(self) -> List[Task]:
        failed: List[Task] = []
        now = self.now()
        for t in self.tasks.values():
            if t.is_open() and now >= t.deadline_ts:
                t.failed_ts = now
                failed.append(t)
        return failed

    def open_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_open()]

    def get(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def contribute(self, *, task_id: str, agent_id: str, agent_role: str) -> Dict[str, object]:
        """
        True teamwork rule:
          - Each agent can contribute at most once per task.
          - Optionally require role match to contribute at all.

        Returns:
          ok: bool
          newly_completed: bool
          role_match: bool
          remaining: int
          already_contributed: bool
          rejected_role: bool
        """
        t = self.tasks.get(task_id)
        if t is None or not t.is_open():
            return {
                "ok": False,
                "newly_completed": False,
                "role_match": False,
                "remaining": 0,
                "already_contributed": False,
                "rejected_role": False,
            }

        if agent_id in t.contributors:
            return {
                "ok": False,
                "newly_completed": False,
                "role_match": agent_role in t.required_roles,
                "remaining": max(0, t.required_contributions - t.contributions),
                "already_contributed": True,
                "rejected_role": False,
            }

        role_match = agent_role in t.required_roles
        if self.require_role_match and not role_match:
            return {
                "ok": False,
                "newly_completed": False,
                "role_match": False,
                "remaining": max(0, t.required_contributions - t.contributions),
                "already_contributed": False,
                "rejected_role": True,
            }

        # accept contribution
        t.contributors.add(agent_id)
        t.contributions += 1

        newly_completed = False
        if t.contributions >= t.required_contributions:
            t.completed_ts = self.now()
            newly_completed = True

        remaining = max(0, t.required_contributions - t.contributions)
        return {
            "ok": True,
            "newly_completed": newly_completed,
            "role_match": role_match,
            "remaining": remaining,
            "already_contributed": False,
            "rejected_role": False,
        }