"""Clean agent execution models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """Represents a single step in the agent's execution."""

    step_id: str
    iteration: int
    step_type: Literal["system_init", "tool_use", "tool_result", "text_response"]
    timestamp: datetime = Field(default_factory=datetime.now)

    # Core content
    content: str

    # Tool-specific fields
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    tool_error: bool = False

    # Metadata
    session_id: str | None = None
    message_id: str | None = None
    usage_stats: dict[str, Any] | None = None
    raw_claude_message: dict[str, Any] | None = None


class ExecutionMetadata(BaseModel):
    """Metadata about the execution environment and results."""

    session_id: str
    model_used: str
    execution_start: datetime
    execution_end: datetime
    total_cost_usd: float | None = None
    total_usage: dict[str, Any] | None = None
    mcp_servers_used: list[str] = []
    tools_available: list[str] = []


class AgentResult(BaseModel):
    """Clean result from running the agent on a task."""

    # Execution status
    success: bool
    final_answer: str
    error_message: str | None = None

    # Execution details
    steps: list[AgentStep]
    total_iterations: int
    execution_time_seconds: float

    # Rich metadata
    metadata: ExecutionMetadata

    @property
    def failed_tool_calls(self) -> int:
        """Count of failed tool calls."""
        return sum(1 for step in self.steps if step.tool_error)

    @property
    def successful_tool_calls(self) -> int:
        """Count of successful tool calls."""
        return sum(
            1
            for step in self.steps
            if step.step_type == "tool_result" and not step.tool_error
        )
