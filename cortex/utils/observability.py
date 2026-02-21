"""Logging for retrieval latency, memory add, feedback."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("cortexos.observability")


def log_retrieval(latency_ms: float, k: int, user_id: Optional[str] = None) -> None:
    logger.info("retrieval latency_ms=%.2f k=%s user_id=%s", latency_ms, k, user_id)


def log_memory_add(memory_id: str, user_id: str) -> None:
    logger.info("memory_add memory_id=%s user_id=%s", memory_id, user_id)


def log_feedback(reward: float, used_count: int) -> None:
    logger.info("feedback reward=%.2f used_count=%s", reward, used_count)
