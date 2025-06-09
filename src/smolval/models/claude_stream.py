"""Models for Claude Code stream-json responses."""

from typing import Any, Literal

from pydantic import BaseModel

from .base import BaseContent, BaseMessage


class ClaudeContentItem(BaseContent):
    """Individual content item in Claude messages."""

    text: str | None = None
    name: str | None = None  # for tool_use
    input: dict[str, Any] | None = None  # for tool_use
    id: str | None = None  # for tool_use/tool_result
    tool_use_id: str | None = None  # for tool_result
    content: str | None = None  # for tool_result
    is_error: bool | None = None  # for tool_result


class ClaudeMessage(BaseModel):
    """Claude message structure."""

    id: str | None = None
    type: str
    role: str | None = None
    model: str | None = None
    content: list[ClaudeContentItem]
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


class ClaudeSystemMessage(BaseMessage):
    """System initialization message."""

    type: Literal["system"]
    subtype: str
    cwd: str | None = None
    tools: list[str] = []
    mcp_servers: list[dict[str, Any]] = []
    model: str | None = None
    permission_mode: str | None = None
    api_key_source: str | None = None


class ClaudeAssistantMessage(BaseMessage):
    """Assistant message with tool calls or text."""

    type: Literal["assistant"]
    message: ClaudeMessage
    parent_tool_use_id: str | None = None


class ClaudeUserMessage(BaseMessage):
    """User message with tool results."""

    type: Literal["user"]
    message: dict[str, Any]
    parent_tool_use_id: str | None = None


class ClaudeResultMessage(BaseMessage):
    """Final result message."""

    type: Literal["result"]
    subtype: str
    result: str | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    is_error: bool = False
    num_turns: int | None = None
    total_cost: float | None = None
    usage: dict[str, Any] | None = None
