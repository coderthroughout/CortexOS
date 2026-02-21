"""Context parser: raw chat, docs, or tool usage -> normalized input for extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Literal, Optional, Union


@dataclass
class ChatTurn:
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


@dataclass
class RawInput:
    """Normalized input for extraction. Built from chat, docs, actions, or tool usage."""

    source: Literal["chat", "doc", "tool"]
    content: str  # concatenated text to send to extractor
    turns: List[ChatTurn] = field(default_factory=list)  # for chat: per-turn
    metadata: dict = field(default_factory=dict)  # session_id, conversation_id, etc.
    timestamp: Optional[datetime] = None


def parse_chat(messages: List[dict], session_id: Optional[str] = None, conversation_id: Optional[str] = None) -> RawInput:
    """
    Normalize chat messages into a single content string and optional turns.
    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
    """
    turns = []
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            # OpenAI-style content blocks
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        turns.append(ChatTurn(role=role, content=content))
        parts.append(f"{role}: {content}")
    return RawInput(
        source="chat",
        content="\n".join(parts),
        turns=turns,
        metadata={"session_id": session_id, "conversation_id": conversation_id},
    )


def parse_document(text: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> RawInput:
    """Normalize a document for extraction."""
    return RawInput(
        source="doc",
        content=text,
        metadata={"doc_id": doc_id, "title": title},
    )


def parse_tool_usage(tool_name: str, input_payload: Any, result: Optional[str] = None) -> RawInput:
    """Normalize tool call + result for extraction."""
    content = f"Tool: {tool_name}\nInput: {input_payload}"
    if result is not None:
        content += f"\nResult: {result}"
    return RawInput(source="tool", content=content, metadata={"tool": tool_name})
