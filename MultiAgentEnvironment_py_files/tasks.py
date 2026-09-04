import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# A "context" is the (arm, feature_vector) pair an agent's policy used to
# make a decision, e.g. ("collab", [1.0, 0.3, ...]). Since GlobalPolicy is
# shared across all agents, any code holding a context can apply a reward
# to it later via policy.update_reactive(arm, x, reward) -- it doesn't need
# to be the same agent object that made the original decision.
Context = Tuple[str, List[float]]


@dataclass
class Invite:
    """
    Records a 'collab' recruitment: `inviter_id` invited `invited_id` to help
    with a task, using decision context `inviter_ctx`. Used to give the
    inviter counterfactual credit if the invite actually leads to a
    contribution (see TaskManager.contribute).
    """

    inviter_id: str
    inviter_ctx: Context
    invited_id: str
    credited: bool = False


@dataclass
class Task:
    """
    A single cooperative task.

    Agents contribute (at most once each) until `required_contributions` is
    reached, or the task expires at `deadline_ts` and is marked failed.
    """

    task_id: str
    created_ts: float
    deadline_ts: float
    required_roles: List[str]
    required_contributions: int

    contributions: int = 0
    contributors: Set[str] = field(default_factory=set)

    # Credit-assignment bookkeeping (see TaskManager.contribute /
    # TaskManager.record_invite for how these get used).
    contributor_contexts: Dict[str, Context] = field(default_factory=dict)
    invites: List[Invite] = field(default_factory=list)

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
    Cooperative task system (this is the "game" the swarm plays):

      - Tasks arrive over time on a fixed schedule (`spawn_times`).
      - Each agent may contribute to a given task at most once (true teamwork,
        not just repeated effort from a single agent).
      - A task completes once enough *unique* contributors have chipped in
        before its deadline; otherwise it fails.
      - `required_roles` is informational/reward-shaping only unless
        `require_role_match=True`, in which case only agents whose role is
        in the list may contribute at all.
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
        require_role_match: bool = False,  # set True for strict role gating
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

        # Pre-schedule when each task spawns during the episode (simple
        # uniform arrival process over the episode duration).
        self.spawn_times = sorted(
            self.rng.uniform(0.0, max(0.1, episode_duration_s - 1.0))
            for _ in range(expected_tasks)
        )
        self._spawn_idx = 0
        self._t0: Optional[float] = None

    def start(self):
        """Mark t=0 for the episode. Must be called before tick_spawn()."""
        self._t0 = time.time()

    def now(self) -> float:
        return time.time()

    def elapsed(self) -> float:
        assert self._t0 is not None
        return self.now() - self._t0

    def tick_spawn(self) -> List[Task]:
        """Spawn any tasks whose scheduled arrival time has passed. Call this each orchestrator tick."""
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
        """Mark any open task past its deadline as failed. Call this each orchestrator tick."""
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

    def record_invite(self, *, task_id: str, inviter_id: str, inviter_ctx: Context, invited_id: str):
        """
        Record that `inviter_id` recruited `invited_id` for a task, using
        decision context `inviter_ctx` (the (arm, x) pair active when the
        inviter chose "collab"). If `invited_id` later contributes to this
        same task, `contribute()` will return this context so the caller
        can credit the inviter for a successful recruitment -- this is the
        counterfactual signal a flat per-contribution reward can't capture.

        No-ops if the task doesn't exist or is already closed.
        """
        t = self.tasks.get(task_id)
        if t is None or not t.is_open():
            return
        t.invites.append(Invite(inviter_id=inviter_id, inviter_ctx=inviter_ctx, invited_id=invited_id))

    def contribute(
        self,
        *,
        task_id: str,
        agent_id: str,
        agent_role: str,
        ctx: Optional[Context] = None,
    ) -> Dict[str, object]:
        """
        Record a contribution attempt from an agent.

        Rules:
          - No effect if the task doesn't exist or is already closed.
          - Each agent may contribute at most once per task.
          - If `require_role_match` is set, only matching roles may contribute.

        `ctx` is the (arm, x) decision context that led to this contribution
        (pass None if you don't want credit-assignment tracking). It's
        stored so that, if this contribution completes the task, the
        completion reward can be split fairly across every contributor
        instead of only going to whichever agent happened to be last.

        Returns a dict with:
          ok:                    whether the contribution was accepted
          newly_completed:       whether this contribution just completed the task
          role_match:            whether the agent's role was in required_roles
          remaining:             contributions still needed
          already_contributed:   True if this agent already contributed before
          rejected_role:         True if rejected due to role mismatch
          required_contributions: task's total required contributions (for splitting reward)
          contributor_contexts:  list of every contributor's (arm, x), only
                                  populated when newly_completed is True
          invite_credit:         the inviter's (arm, x) context, if this
                                  contribution fulfills a pending invite --
                                  else None
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
                "required_contributions": t.required_contributions if t else 0,
                "contributor_contexts": [],
                "invite_credit": None,
            }

        if agent_id in t.contributors:
            return {
                "ok": False,
                "newly_completed": False,
                "role_match": agent_role in t.required_roles,
                "remaining": max(0, t.required_contributions - t.contributions),
                "already_contributed": True,
                "rejected_role": False,
                "required_contributions": t.required_contributions,
                "contributor_contexts": [],
                "invite_credit": None,
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
                "required_contributions": t.required_contributions,
                "contributor_contexts": [],
                "invite_credit": None,
            }

        t.contributors.add(agent_id)
        t.contributions += 1
        if ctx is not None:
            t.contributor_contexts[agent_id] = ctx

        # Counterfactual recruitment credit: if someone invited this agent
        # to this task and hasn't been credited yet, hand back their context.
        invite_credit: Optional[Context] = None
        for inv in t.invites:
            if inv.invited_id == agent_id and not inv.credited:
                inv.credited = True
                invite_credit = inv.inviter_ctx
                break

        newly_completed = False
        contributor_contexts: List[Context] = []
        if t.contributions >= t.required_contributions:
            t.completed_ts = self.now()
            newly_completed = True
            # Every recorded contributor context gets a share of the
            # completion reward -- see module docstring / agent.py for why
            # this replaces an all-to-the-last-contributor bonus.
            contributor_contexts = list(t.contributor_contexts.values())

        remaining = max(0, t.required_contributions - t.contributions)
        return {
            "ok": True,
            "newly_completed": newly_completed,
            "role_match": role_match,
            "remaining": remaining,
            "already_contributed": False,
            "rejected_role": False,
            "required_contributions": t.required_contributions,
            "contributor_contexts": contributor_contexts,
            "invite_credit": invite_credit,
        }