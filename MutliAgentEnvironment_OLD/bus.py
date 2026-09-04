import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Message:
    message_id: str
    conversation_id: str
    from_agent_id: str
    to_agent_id: Optional[str]  # None means broadcast
    kind: str
    body: Dict[str, Any]


class InMemoryBus:
    """
    Simple message bus:
      - each agent has an asyncio.Queue
      - supports direct send and broadcast
    """

    def __init__(self):
        self.queues: Dict[str, asyncio.Queue[Message]] = {}

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        q: asyncio.Queue[Message] = asyncio.Queue()
        self.queues[agent_id] = q
        return q

    def unregister_agent(self, agent_id: str):
        self.queues.pop(agent_id, None)

    async def send(self, msg: Message):
        if msg.to_agent_id is None:
            # broadcast to everyone except sender
            for aid, q in self.queues.items():
                if aid != msg.from_agent_id:
                    await q.put(msg)
        else:
            q = self.queues.get(msg.to_agent_id)
            if q is not None:
                await q.put(msg)

    @staticmethod
    def new_message(
        from_agent_id: str,
        to_agent_id: Optional[str],
        kind: str,
        body: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> Message:
        return Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id or str(uuid.uuid4()),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            kind=kind,
            body=body,
        )