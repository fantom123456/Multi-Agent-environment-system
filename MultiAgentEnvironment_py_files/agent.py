import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from bus import InMemoryBus, Message
from policy import GlobalPolicy
from telemetry import Telemetry, new_event
from tasks import TaskManager


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    seed: int

    # --- RESERVED for a future proactive-mode agent (currently unused) ---
    # These are still populated by orchestrator.py for forward-compatibility,
    # but nothing in this file reads them: proactive behavior is disabled
    # (see Agent._maybe_proactive below), so an agent never decides on its
    # own to ping or broadcast — it only reacts to incoming task messages.
    proactive_interval_s: float = 1.5
    broadcast_probability: float = 0.25
    reply_probability: float = 0.7
    collaboration_probability: float = 0.2


@dataclass
class AgentState:
    inbox_count: int = 0
    sent_count: int = 0
    known_agents: list[str] = field(default_factory=list)
    # RESERVED: only meaningful once proactive mode is re-enabled — it never
    # gets updated today, so `seconds_since_proactive` in featurize() is a
    # near-constant (see Agent._features).
    last_proactive_ts: float = 0.0
    last_received_kind: Optional[str] = None


class Agent:
    """
    Tasks-only agent.

    Behavior:
      - Only reacts to `task_request` / `task_invite` messages; every other
        message kind is received (for telemetry) but otherwise ignored.
      - Proactive behavior (an agent initiating a ping/broadcast on its own)
        is disabled — tasks are always pushed by the orchestrator.
      - On a task message, the shared LinUCB policy (`GlobalPolicy`) chooses
        one of {ignore, reply, collab}, the agent attempts to contribute to
        the task via `TaskManager`, and the outcome is turned into a reward
        that's fed back into the policy.

    Credit assignment (see tasks.py for the underlying bookkeeping):
      - Per-contribution progress reward goes to whoever contributed, as before.
      - Completion reward is split evenly across every contributor's own
        decision context (fair for a quota-based task, where each of the
        required contributors is equally pivotal), rather than all going to
        whichever agent happened to complete the quota last.
      - A "collab" invite's decision context gets a small counterfactual
        bonus if the invited peer actually goes on to contribute -- giving
        the recruiting decision credit for outcomes beyond the recruiter's
        own immediate contribution.

    RESERVED / not currently active:
      - `proactive_interval_s` (constructor arg) and the proactive-mode
        AgentConfig fields above are kept so a future "proactive" agent mode
        (e.g. agents that bid on or discover tasks themselves) can reuse
        this class without a redesign.
      - `TASKS_ONLY` below is a documentation flag, not a live switch: the
        "ignore anything that's not task-related" behavior in
        `_handle_message` is hardcoded rather than gated by this attribute.
        If you want it to actually toggle behavior, check `self.TASKS_ONLY`
        in `_handle_message` instead of hardcoding the kind check.
    """

    def __init__(
        self,
        run_id: str,
        cfg: AgentConfig,
        bus: InMemoryBus,
        inbox: asyncio.Queue,
        telemetry: Telemetry,
        all_agent_ids: list[str],
        policy: GlobalPolicy,
        proactive_interval_s: float = 1.0,  # RESERVED, unused — see class docstring
        task_manager: Optional[TaskManager] = None,
    ):
        self.run_id = run_id
        self.cfg = cfg
        self.bus = bus
        self.inbox = inbox
        self.telemetry = telemetry
        self.policy = policy
        self.proactive_interval_s = proactive_interval_s
        self.task_manager = task_manager

        self.rng = random.Random(cfg.seed)
        self.state = AgentState(known_agents=[a for a in all_agent_ids if a != cfg.agent_id])

        self._last_proactive_ctx: Optional[tuple[str, list[float]]] = None  # RESERVED, unused
        self._last_reactive_ctx: Optional[tuple[str, list[float]]] = None   # ACTIVE

        self._stop = asyncio.Event()

        # Reward shaping for task completion + anti-spam.
        self.R_TASK_PROGRESS = 0.2
        self.R_TASK_COMPLETE = 10.0

        self.C_SEND = 0.02
        self.C_BROADCAST_EXTRA = 0.08  # only applies if this agent ever sends a "broadcast"-kind message
        self.C_STEP = 0.001

        # Counterfactual recruitment credit: rewarded to a "collab" decision's
        # own context if the peer it invited actually goes on to contribute.
        # See tasks.py's Invite/record_invite for how this is tracked.
        self.R_INVITE_SUCCESS = 1.0

        # Documentation flag only — see class docstring. Not read anywhere.
        self.TASKS_ONLY = True

    async def stop(self):
        self._stop.set()

    def _pick_peer(self) -> Optional[str]:
        return self.rng.choice(self.state.known_agents) if self.state.known_agents else None

    def _features(self, seconds_since_proactive: float, last_msg_kind: Optional[str]) -> list[float]:
        return self.policy.featurize(
            inbox_count=self.state.inbox_count,
            sent_count=self.state.sent_count,
            known_agents=len(self.state.known_agents),
            seconds_since_proactive=seconds_since_proactive,
            last_msg_kind=last_msg_kind,
        )

    async def _log_agent_started(self):
        await self.telemetry.log(
            new_event(
                run_id=self.run_id,
                event_type="agent_started",
                agent_id=self.cfg.agent_id,
                payload={"role": self.cfg.role, "seed": self.cfg.seed},
            )
        )

    def _apply_reward_to_last(self, reward: float):
        """
        Best-effort credit assignment: apply `reward` to whichever decision
        this agent made most recently. In tasks-only mode `_last_reactive_ctx`
        is always the one that's set — the proactive branch below is dead
        code today but kept for when proactive mode returns.
        """
        if self._last_reactive_ctx is not None:
            arm, x = self._last_reactive_ctx
            self.policy.update_reactive(arm, x, reward)
        elif self._last_proactive_ctx is not None:  # RESERVED / currently unreachable
            arm, x = self._last_proactive_ctx
            self.policy.update_proactive(arm, x, reward)

    def _reward_send_cost(self, msg_kind: str):
        r = -self.C_SEND
        if msg_kind == "broadcast":
            r -= self.C_BROADCAST_EXTRA
        self._apply_reward_to_last(r)

    async def _send(self, msg: Message):
        await self.telemetry.log(
            new_event(
                run_id=self.run_id,
                event_type="message_sent",
                agent_id=self.cfg.agent_id,
                from_agent_id=msg.from_agent_id,
                to_agent_id=msg.to_agent_id,
                conversation_id=msg.conversation_id,
                message_id=msg.message_id,
                payload={"kind": msg.kind, "body": msg.body},
            )
        )
        await self.bus.send(msg)
        self.state.sent_count += 1
        self._reward_send_cost(msg.kind)

    async def _handle_task_message(self, msg: Message):
        """Core decision loop: choose ignore/reply/collab, act on it, and reward the outcome."""
        if self.task_manager is None:
            return

        task_id = msg.body.get("task_id")
        if not task_id:
            return

        seconds_since_proactive = time.time() - self.state.last_proactive_ts
        x = self._features(seconds_since_proactive, last_msg_kind=msg.kind)

        arm = self.policy.choose_reactive(x)
        ctx = (arm, x)
        self._last_reactive_ctx = ctx

        # Small per-step cost, applied regardless of the chosen action, so the
        # policy has a signal even when it chooses "ignore".
        self.policy.update_reactive(arm, x, -self.C_STEP)

        if arm == "ignore":
            return

        # Attempt to contribute to the task (reply and collab both contribute;
        # collab additionally recruits a peer below). `ctx` is passed through
        # so TaskManager can hand back fair completion/invite credit later.
        outcome = self.task_manager.contribute(
            task_id=task_id,
            agent_id=self.cfg.agent_id,
            agent_role=self.cfg.role,
            ctx=ctx,
        )

        if outcome.get("ok"):
            r = self.R_TASK_PROGRESS * (2.0 if outcome.get("role_match") else 1.0)
            self.policy.update_reactive(arm, x, r)

            await self.telemetry.log(
                new_event(
                    run_id=self.run_id,
                    event_type="task_contributed",
                    agent_id=self.cfg.agent_id,
                    payload={
                        "task_id": task_id,
                        "role": self.cfg.role,
                        "role_match": outcome.get("role_match"),
                        "remaining": outcome.get("remaining"),
                    },
                )
            )

            # Counterfactual recruitment credit: if this contribution fulfills
            # a pending "collab" invite, reward the inviter's original
            # decision context -- this is the signal a flat per-contribution
            # reward can't capture (was the invite actually what caused this
            # contribution to happen?).
            invite_credit = outcome.get("invite_credit")
            if invite_credit is not None:
                inviter_arm, inviter_x = invite_credit
                self.policy.update_reactive(inviter_arm, inviter_x, self.R_INVITE_SUCCESS)
                await self.telemetry.log(
                    new_event(
                        run_id=self.run_id,
                        event_type="invite_credited",
                        agent_id=self.cfg.agent_id,
                        payload={"task_id": task_id},
                    )
                )

            if outcome.get("newly_completed"):
                # Fair completion credit: split R_TASK_COMPLETE evenly across
                # every contributor's own decision context, since in a
                # quota-based task each contributor up to the threshold is
                # equally pivotal (removing any one would have meant the task
                # needed one more contributor). This replaces the old
                # behavior of handing the entire bonus to whichever agent
                # happened to be last -- a pure timing artifact, not a
                # measure of who actually mattered.
                required = outcome.get("required_contributions") or 1
                share = self.R_TASK_COMPLETE / max(1, required)
                for c_arm, c_x in outcome.get("contributor_contexts", []):
                    self.policy.update_reactive(c_arm, c_x, share)

                await self.telemetry.log(
                    new_event(
                        run_id=self.run_id,
                        event_type="task_completed",
                        payload={"task_id": task_id, "completed_by": self.cfg.agent_id},
                    )
                )

        if arm == "collab":
            third = self._pick_peer()
            if third:
                invite = self.bus.new_message(
                    from_agent_id=self.cfg.agent_id,
                    to_agent_id=third,
                    kind="task_invite",
                    conversation_id=msg.conversation_id,
                    body={"task_id": task_id, "text": "Please contribute if you can."},
                )
                await self._send(invite)
                # Record the invite so a later contribution from `third` can
                # credit this agent's collab decision (ctx captured above).
                self.task_manager.record_invite(
                    task_id=task_id,
                    inviter_id=self.cfg.agent_id,
                    inviter_ctx=ctx,
                    invited_id=third,
                )

    async def _handle_message(self, msg: Message):
        self.state.inbox_count += 1
        self.state.last_received_kind = msg.kind

        await self.telemetry.log(
            new_event(
                run_id=self.run_id,
                event_type="message_received",
                agent_id=self.cfg.agent_id,
                from_agent_id=msg.from_agent_id,
                to_agent_id=msg.to_agent_id,
                conversation_id=msg.conversation_id,
                message_id=msg.message_id,
                payload={"kind": msg.kind, "body": msg.body},
            )
        )

        # Tasks-only: everything except task_request/task_invite is logged
        # above (for telemetry) and then ignored.
        if msg.kind not in ("task_request", "task_invite"):
            return

        await self._handle_task_message(msg)

    async def _maybe_proactive(self):
        """
        RESERVED extension point, currently a no-op.

        Proactive behavior is disabled because tasks are always pushed by
        the orchestrator. If you add agent-initiated behavior later (e.g.
        an agent proactively asking around for open tasks, or bidding),
        implement it here using `self.policy.choose_proactive(...)` and
        record the choice in `self._last_proactive_ctx` so
        `_apply_reward_to_last` can credit it.
        """
        return

    async def run(self):
        await self._log_agent_started()

        while not self._stop.is_set():
            try:
                msg = await asyncio.wait_for(self.inbox.get(), timeout=0.2)
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                pass

            await self._maybe_proactive()

        await self.telemetry.log(
            new_event(
                run_id=self.run_id,
                event_type="agent_stopped",
                agent_id=self.cfg.agent_id,
                payload={"inbox_count": self.state.inbox_count, "sent_count": self.state.sent_count},
            )
        )