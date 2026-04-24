"""
memory.py — Conversation Memory Manager.

Stores recent user messages and assistant responses in a sliding window.
Provides formatted context for prompt injection.

Design:
  - Each conversation has a unique session_id.
  - Stores (role, content) tuples in order.
  - get_context(n) returns the last n exchanges as a formatted string.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationTurn:
    """Represents a single turn in the conversation."""

    def __init__(self, role: str, content: str):
        """
        Args:
            role:    'user' or 'assistant'
            content: The message text.
        """
        self.role      = role
        self.content   = content
        self.timestamp = datetime.utcnow().isoformat()

    def __repr__(self):
        return f"ConversationTurn(role={self.role!r}, content={self.content[:40]!r})"


class ConversationMemory:
    """
    Sliding-window conversation memory for a single session.

    Attributes:
        session_id:   Unique identifier for this conversation.
        max_turns:    Maximum number of turns to retain (default: 10).
        history:      deque of ConversationTurn objects.
    """

    def __init__(self, session_id: str, max_turns: int = 10):
        self.session_id = session_id
        self.max_turns  = max_turns
        # deque auto-drops oldest entries when maxlen is reached
        self.history: deque[ConversationTurn] = deque(maxlen=max_turns)
        logger.debug(f"[Memory] Session created: {session_id}")

    def add_user_message(self, content: str) -> None:
        """Append a user message to history."""
        turn = ConversationTurn(role="user", content=content)
        self.history.append(turn)
        logger.debug(f"[Memory] +User: {content[:60]}")

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to history."""
        turn = ConversationTurn(role="assistant", content=content)
        self.history.append(turn)
        logger.debug(f"[Memory] +Assistant: {content[:60]}")

    def get_context(self, last_n: int = 4) -> str:
        """
        Return the last `last_n` turns as a formatted context string
        ready for injection into the LLM prompt.

        Format:
            User: <message>
            Assistant: <response>
            ...

        Args:
            last_n: Number of recent turns to include.

        Returns:
            Multi-line context string, or empty string if no history.
        """
        if not self.history:
            return ""

        # Grab the last `last_n` items from the deque
        recent = list(self.history)[-last_n:]
        lines  = []

        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")

        context = "\n".join(lines)
        logger.debug(f"[Memory] Context ({len(recent)} turns):\n{context[:200]}")
        return context

    def get_all_turns(self) -> list[dict]:
        """
        Return full history as a list of dicts (useful for API responses/logging).
        """
        return [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            for t in self.history
        ]

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()
        logger.info(f"[Memory] Session {self.session_id} cleared.")

    def __len__(self) -> int:
        return len(self.history)


class MemoryStore:
    """
    Global store of ConversationMemory objects keyed by session_id.

    Usage:
        store = MemoryStore()
        mem   = store.get_or_create("session-abc")
        mem.add_user_message("Hello!")
    """

    def __init__(self, max_turns_per_session: int = 10):
        self._sessions: dict[str, ConversationMemory] = {}
        self._max_turns = max_turns_per_session

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """
        Retrieve existing session memory or create a new one.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationMemory instance.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                session_id=session_id,
                max_turns=self._max_turns,
            )
            logger.info(f"[MemoryStore] New session: {session_id}")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from the store."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"[MemoryStore] Deleted session: {session_id}")
            return True
        return False

    def active_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())
