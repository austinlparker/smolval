"""Models for smolval data structures."""

from .agent import AgentResult, AgentStep, ExecutionMetadata
from .claude_stream import (
    ClaudeAssistantMessage,
    ClaudeContentItem,
    ClaudeMessage,
    ClaudeResultMessage,
    ClaudeSystemMessage,
    ClaudeUserMessage,
)

__all__ = [
    # Agent models
    "AgentStep",
    "AgentResult",
    "ExecutionMetadata",
    # Claude stream models
    "ClaudeContentItem",
    "ClaudeMessage",
    "ClaudeSystemMessage",
    "ClaudeAssistantMessage",
    "ClaudeUserMessage",
    "ClaudeResultMessage",
]
