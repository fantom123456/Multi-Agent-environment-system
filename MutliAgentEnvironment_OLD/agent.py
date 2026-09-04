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
    # kept for compatibility
    proactive_interval_s: float = 1.5
    broadcast_probability: float = 0.25
    reply_probability: float = 0.7
    collaboration_probability: float = 0.2


@dataclass
class AgentState:
    inbox_count: int = 0
    sent_count: int = 0
    known_agents: list[str] = field(default_factory=list)
    last_proactive_ts: float = 0.0
    last_received_kind: Optional[str] = None


class Agent:
    """
    TASKS-ONLY agent.

    - Only cares about task_request / task_invite.
    - Ignores all other message kinds to eliminate runaway chatter/spam.
    - Proactive behavior is disabled (noop) because tasks are pushed by orchestrator.
    - Uses the learned reactive policy to decide ignore/reply/collab on task messages.
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
        proactive_interval_s: float = 1.0,
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

        self._last_proactive_ctx: Optional[tuple[str, list[float]]] = None
        self._last_reactive_ctx: Optional[tuple[str, list[float]]] = None

        self._stop = asyncio.Event()

        # Reward shaping for Goal C (task completion) + anti-spam
        self.R_TASK_PROGRESS = 0.2
        self.R_TASK_COMPLETE = 10.0

        self.C_SEND = 0.02
        self.C_BROADCAST_EXTRA = 0.08
        self.C_STEP = 0.001

        # Tasks-only mode toggle (in case you want to re-enable later)
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
        # Best-effort credit assignment
        if self._last_reactive_ctx is not None:
            arm, x = self._last_reactive_ctx
            self.policy.update_reactive(arm, x, reward)
        elif self._last_proactive_ctx is not None:
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
        if self.task_manager is None:
            return

        task_id = msg.body.get("task_id")
        if not task_id:
            return

        seconds_since_proactive = time.time() - self.state.last_proactive_ts
        x = self._features(seconds_since_proactive, last_msg_kind=msg.kind)

        # Decide how to react to task request
        arm = self.policy.choose_reactive(x)
        self._last_reactive_ctx = (arm, x)

        # Step cost (forces updates even if ignore)
        self.policy.update_reactive(arm, x, -self.C_STEP)

        if arm == "ignore":
            return

        # Attempt contribution
        outcome = self.task_manager.contribute(
            task_id=task_id,
            agent_id=self.cfg.agent_id,
            agent_role=self.cfg.role,
        )

        if outcome.get("ok"):
            # Progress reward, boosted for role-match
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

            if outcome.get("newly_completed"):
                self.policy.update_reactive(arm, x, self.R_TASK_COMPLETE)
                await self.telemetry.log(
                    new_event(
                        run_id=self.run_id,
                        event_type="task_completed",
                        payload={"task_id": task_id, "completed_by": self.cfg.agent_id},
                    )
                )

        # Collab means: recruit one more agent to help
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

        # TASKS-ONLY: ignore anything that's not task-related
        if msg.kind not in ("task_request", "task_invite"):
            return

        await self._handle_task_message(msg)

    async def _maybe_proactive(self):
        # Tasks-only: proactive disabled to prevent non-task spam.
        # If you later add "task discovery" or "task bidding", you can re-enable
        # a task-focused proactive action here.
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