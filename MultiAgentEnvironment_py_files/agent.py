import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from bus import InMemoryBus, Message
from policy import GlobalPolicy
from telemetry import Telemetry, new_event


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    seed: int
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
    def __init__(
        self,
        run_id: str,
        cfg: AgentConfig,
        bus: InMemoryBus,
        inbox: asyncio.Queue,
        telemetry: Telemetry,
        all_agent_ids: list[str],
        policy: GlobalPolicy,
        all_agent_roles: Optional[dict[str, str]] = None,
        proactive_interval_s: float = 1.0,
    ):
        self.run_id = run_id
        self.cfg = cfg
        self.bus = bus
        self.inbox = inbox
        self.telemetry = telemetry
        self.policy = policy
        self.proactive_interval_s = proactive_interval_s
        self.agent_roles: dict[str, str] = all_agent_roles or {}

        self.rng = random.Random(cfg.seed)
        self.state = AgentState(known_agents=[a for a in all_agent_ids if a != cfg.agent_id])

        self._last_proactive_ctx: Optional[tuple[str, list[float]]] = None
        self._last_reactive_ctx: Optional[tuple[str, list[float]]] = None

        self._stop = asyncio.Event()

        self.R_TASK_PROGRESS = config.R_TASK_PROGRESS
        self.R_TASK_COMPLETE = config.R_TASK_COMPLETE
        self.C_SEND = config.C_SEND
        self.C_BROADCAST_EXTRA = config.C_BROADCAST_EXTRA
        self.C_STEP = config.C_STEP
        self.R_INVITE_SUCCESS = config.R_INVITE_SUCCESS

    async def stop(self):
        self._stop.set()

    def _pick_peer(self, required_roles: Optional[list[str]] = None) -> Optional[str]:
        if not self.state.known_agents:
            return None

        if required_roles and self.agent_roles:
            matching = [a for a in self.state.known_agents if self.agent_roles.get(a) in required_roles]
            if matching:
                return self.rng.choice(matching)

        return self.rng.choice(self.state.known_agents)

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
        task_id = msg.body.get("task_id")
        if not task_id:
            return

        seconds_since_proactive = time.time() - self.state.last_proactive_ts
        x = self._features(seconds_since_proactive, last_msg_kind=msg.kind)

        arm = self.policy.choose_reactive(x)
        ctx = (arm, x)
        self._last_reactive_ctx = ctx

        self.policy.update_reactive(arm, x, -self.C_STEP)

        if arm == "ignore":
            return

        # Explicit decoupled contribution intent sent over network
        contrib_msg = self.bus.new_message(
            from_agent_id=self.cfg.agent_id,
            to_agent_id="system",
            kind="task_contribute",
            conversation_id=msg.conversation_id,
            body={"task_id": task_id, "role": self.cfg.role, "ctx": ctx},
        )
        await self._send(contrib_msg)

        if arm == "collab":
            req_roles = msg.body.get("required_roles")
            third = self._pick_peer(req_roles)
            if third:
                invite = self.bus.new_message(
                    from_agent_id=self.cfg.agent_id,
                    to_agent_id=third,
                    kind="task_invite",
                    conversation_id=msg.conversation_id,
                    body={
                        "task_id": task_id,
                        "required_roles": req_roles,
                        "text": "Please contribute if you can.",
                    },
                )
                await self._send(invite)

                # Record invite event across system bus
                invite_rec = self.bus.new_message(
                    from_agent_id=self.cfg.agent_id,
                    to_agent_id="system",
                    kind="task_invite_record",
                    conversation_id=msg.conversation_id,
                    body={"task_id": task_id, "invited_id": third, "ctx": ctx},
                )
                await self._send(invite_rec)

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

        # Asynchronous acknowledgments and credits received over bus
        if msg.kind == "task_contrib_ack":
            arm, x = msg.body["ctx"]
            self.policy.update_reactive(arm, x, msg.body["reward"])
            return

        if msg.kind == "invite_credit_notice":
            arm, x = msg.body["ctx"]
            self.policy.update_reactive(arm, x, self.R_INVITE_SUCCESS)
            await self.telemetry.log(
                new_event(
                    run_id=self.run_id,
                    event_type="invite_credited",
                    agent_id=self.cfg.agent_id,
                    payload={"task_id": msg.body["task_id"]},
                )
            )
            return

        if msg.kind == "task_complete_notice":
            arm, x = msg.body["ctx"]
            self.policy.update_reactive(arm, x, msg.body["share"])
            return

        if msg.kind in ("task_request", "task_invite"):
            await self._handle_task_message(msg)

    async def run(self):
        await self._log_agent_started()

        while not self._stop.is_set():
            try:
                msg = await asyncio.wait_for(self.inbox.get(), timeout=0.2)
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                pass

        await self.telemetry.log(
            new_event(
                run_id=self.run_id,
                event_type="agent_stopped",
                agent_id=self.cfg.agent_id,
                payload={"inbox_count": self.state.inbox_count, "sent_count": self.state.sent_count},
            )
        )