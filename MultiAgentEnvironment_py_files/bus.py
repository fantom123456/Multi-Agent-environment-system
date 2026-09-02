import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Message:
    """A single message passed between agents (or from the orchestrator/'system')."""

    message_id: str
    conversation_id: str
    from_agent_id: str
    to_agent_id: Optional[str]  # None means broadcast to every other agent
    kind: str
    body: Dict[str, Any]


class InMemoryBus:
    """
    Minimal async message bus.

    Each agent gets its own asyncio.Queue (registered via `register_agent`).
    `send()` either delivers to one agent (direct) or to everyone except the
    sender (broadcast, when `to_agent_id` is None).
    """

    def __init__(self):
        self.queues: Dict[str, asyncio.Queue[Message]] = {}

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        """Create and return the inbox queue for a new agent."""
        q: asyncio.Queue[Message] = asyncio.Queue()
        self.queues[agent_id] = q
        return q

    def unregister_agent(self, agent_id: str):
        """Remove an agent's inbox (e.g. on shutdown)."""
        self.queues.pop(agent_id, None)

    async def send(self, msg: Message):
        """Route a message to its recipient, or to all agents if it's a broadcast."""
        if msg.to_agent_id is None:
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
        """Convenience constructor that fills in message_id / conversation_id."""
        return Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id or str(uuid.uuid4()),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            kind=kind,
            body=body,
        )