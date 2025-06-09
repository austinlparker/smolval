"""Modern parser for Claude Code stream-json output."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from .models import (
    AgentResult,
    AgentStep,
    ClaudeAssistantMessage,
    ClaudeResultMessage,
    ClaudeSystemMessage,
    ClaudeUserMessage,
    ExecutionMetadata,
)

logger = logging.getLogger(__name__)


class ClaudeStreamParser:
    """Modern parser for Claude Code stream-json output."""

    def __init__(self) -> None:
        self.execution_start = datetime.now()

    def parse_stream(self, output: str) -> AgentResult:
        """Parse Claude stream output into a complete AgentResult."""
        if not output.strip():
            return self._create_error_result("No output received from Claude Code")

        lines = output.strip().split("\n")
        steps: list[AgentStep] = []
        final_answer = ""
        metadata_info = {}
        session_id = ""

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            try:
                parsed_message = self._parse_json_line(line)
                if parsed_message:
                    step, answer, meta = self._process_message(
                        parsed_message, len(steps) + 1
                    )

                    if step:
                        steps.append(step)
                    if answer:
                        final_answer = answer
                    if meta:
                        metadata_info.update(meta)
                        if not session_id and meta.get("session_id"):
                            session_id = meta["session_id"]

            except Exception as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

        execution_end = datetime.now()
        execution_time = (execution_end - self.execution_start).total_seconds()

        # Create metadata
        metadata = ExecutionMetadata(
            session_id=session_id or str(uuid.uuid4()),
            model_used=metadata_info.get("model", "unknown"),
            execution_start=self.execution_start,
            execution_end=execution_end,
            total_cost_usd=metadata_info.get("total_cost"),
            total_usage=metadata_info.get("usage"),
            mcp_servers_used=metadata_info.get("mcp_servers", []),
            tools_available=metadata_info.get("tools", []),
        )

        return AgentResult(
            success=bool(final_answer and not metadata_info.get("is_error", False)),
            final_answer=final_answer or "No final answer received",
            error_message=metadata_info.get("error_message"),
            steps=steps,
            total_iterations=len(steps),
            execution_time_seconds=execution_time,
            metadata=metadata,
        )

    def _parse_json_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single JSON line."""
        try:
            parsed = json.loads(line)
            # Ensure we return a dict if json.loads succeeds
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON line: {e}")
            return None

    def _process_message(
        self, data: dict[str, Any], iteration: int
    ) -> tuple[AgentStep | None, str, dict[str, Any]]:
        """Process a Claude message into step, answer, and metadata."""
        msg_type = data.get("type")

        if not isinstance(msg_type, str):
            return None, "", {}

        processors = {
            "system": self._process_system_message,
            "assistant": self._process_assistant_message,
            "user": self._process_user_message,
            "result": self._process_result_message,
        }

        processor = processors.get(msg_type)
        if processor:
            return processor(data, iteration)

        return None, "", {}

    def _process_system_message(
        self, data: dict[str, Any], iteration: int
    ) -> tuple[AgentStep | None, str, dict[str, Any]]:
        """Process system initialization message."""
        try:
            system_msg = ClaudeSystemMessage(**data)

            # Create detailed content about the initialization
            content_parts = [
                f"Claude Code initialized (model: {system_msg.model or 'unknown'})"
            ]

            if system_msg.cwd:
                content_parts.append(f"Working directory: {system_msg.cwd}")

            if system_msg.tools:
                content_parts.append(
                    f"Available tools ({len(system_msg.tools)}): {', '.join(system_msg.tools)}"
                )

            if system_msg.mcp_servers:
                server_names = []
                for server in system_msg.mcp_servers:
                    name = server.get("name", "unknown")
                    command = server.get("command", "")
                    if command:
                        server_names.append(f"{name} ({command})")
                    else:
                        server_names.append(name)
                content_parts.append(
                    f"MCP servers ({len(system_msg.mcp_servers)}): {', '.join(server_names)}"
                )

            if system_msg.permission_mode:
                content_parts.append(f"Permission mode: {system_msg.permission_mode}")

            content = "\n".join(content_parts)

            step = AgentStep(
                step_id=f"system_{iteration}",
                iteration=iteration,
                step_type="system_init",
                content=content,
                session_id=system_msg.session_id,
                raw_claude_message=data,
            )

            metadata = {
                "session_id": system_msg.session_id,
                "model": system_msg.model,
                "tools": system_msg.tools,
                "mcp_servers": [
                    server.get("name", "unknown") for server in system_msg.mcp_servers
                ],
                "working_directory": system_msg.cwd,
                "permission_mode": system_msg.permission_mode,
            }

            return step, "", metadata

        except Exception as e:
            logger.warning(f"Failed to process system message: {e}")
            return None, "", {}

    def _process_assistant_message(
        self, data: dict[str, Any], iteration: int
    ) -> tuple[AgentStep | None, str, dict[str, Any]]:
        """Process assistant message with tool calls or text."""
        try:
            assistant_msg = ClaudeAssistantMessage(**data)

            for content_item in assistant_msg.message.content:
                if content_item.type == "tool_use":
                    step = AgentStep(
                        step_id=content_item.id or f"tool_use_{iteration}",
                        iteration=iteration,
                        step_type="tool_use",
                        content=f"Using tool: {content_item.name}",
                        tool_name=content_item.name,
                        tool_input=content_item.input,
                        session_id=assistant_msg.session_id,
                        message_id=assistant_msg.message.id,
                        usage_stats=assistant_msg.message.usage,
                        raw_claude_message=data,
                    )
                    return step, "", {}

                elif content_item.type == "text" and content_item.text:
                    step = AgentStep(
                        step_id=f"text_{iteration}",
                        iteration=iteration,
                        step_type="text_response",
                        content=content_item.text,
                        session_id=assistant_msg.session_id,
                        message_id=assistant_msg.message.id,
                        usage_stats=assistant_msg.message.usage,
                        raw_claude_message=data,
                    )
                    return step, "", {}

        except Exception as e:
            logger.warning(f"Failed to process assistant message: {e}")

        return None, "", {}

    def _process_user_message(
        self, data: dict[str, Any], iteration: int
    ) -> tuple[AgentStep | None, str, dict[str, Any]]:
        """Process user message with tool results."""
        try:
            user_msg = ClaudeUserMessage(**data)

            content_list = user_msg.message.get("content", [])
            for content_item in content_list:
                if content_item.get("type") == "tool_result":
                    raw_tool_content = content_item.get("content", "")
                    is_error = content_item.get("is_error", False)

                    # Handle both string and list content from Claude
                    if isinstance(raw_tool_content, list):
                        # Extract text from list of content items
                        tool_content = ""
                        for item in raw_tool_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                tool_content += item.get("text", "")
                            elif isinstance(item, str):
                                tool_content += item
                        if not tool_content:
                            tool_content = str(
                                raw_tool_content
                            )  # Fallback to string representation
                    else:
                        tool_content = (
                            str(raw_tool_content)
                            if raw_tool_content is not None
                            else ""
                        )

                    step = AgentStep(
                        step_id=content_item.get(
                            "tool_use_id", f"tool_result_{iteration}"
                        ),
                        iteration=iteration,
                        step_type="tool_result",
                        content=f"Tool result: {tool_content[:200]}{'...' if len(tool_content) > 200 else ''}",
                        tool_output=tool_content,
                        tool_error=is_error,
                        session_id=user_msg.session_id,
                        raw_claude_message=data,
                    )
                    return step, "", {}

        except Exception as e:
            logger.warning(f"Failed to process user message: {e}")

        return None, "", {}

    def _process_result_message(
        self, data: dict[str, Any], iteration: int
    ) -> tuple[AgentStep | None, str, dict[str, Any]]:
        """Process final result message."""
        try:
            result_msg = ClaudeResultMessage(**data)

            final_answer = result_msg.result or ""
            metadata = {
                "total_cost": result_msg.total_cost,
                "usage": result_msg.usage,
                "is_error": result_msg.is_error,
            }

            return None, final_answer, metadata

        except Exception as e:
            logger.warning(f"Failed to process result message: {e}")

        return None, "", {}

    def _create_error_result(self, error_message: str) -> AgentResult:
        """Create an error result."""
        metadata = ExecutionMetadata(
            session_id=str(uuid.uuid4()),
            model_used="unknown",
            execution_start=self.execution_start,
            execution_end=datetime.now(),
        )

        return AgentResult(
            success=False,
            final_answer="",
            error_message=error_message,
            steps=[],
            total_iterations=0,
            execution_time_seconds=0.0,
            metadata=metadata,
        )
