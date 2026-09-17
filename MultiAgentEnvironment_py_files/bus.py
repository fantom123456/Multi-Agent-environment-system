import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Message:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    kind: str
    conversation_id: str
    body: Dict[str, Any] = field(default_factory=dict)


class InMemoryBus:
    def __init__(self):
        self._inboxes: Dict[str, asyncio.Queue] = {}

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue()
        return self._inboxes[agent_id]

    def register_system(self) -> asyncio.Queue:
        return self.register_agent("system")

    def new_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        kind: str,
        conversation_id: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Message:
        return Message(
            message_id=str(uuid.uuid4()),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            kind=kind,
            conversation_id=conversation_id or str(uuid.uuid4()),
            body=body or {},
        )

    async def send(self, msg: Message):
        if msg.to_agent_id == "*":
            for aid, q in self._inboxes.items():
                if aid not in (msg.from_agent_id, "system"):
                    await q.put(msg)
        else:
            q = self._inboxes.get(msg.to_agent_id)
            if q is not None:
                await q.put(msg)