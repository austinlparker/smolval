"""ReAct agent implementation for MCP server evaluation."""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import tiktoken
from pydantic import BaseModel

from smolval.config import Config
from smolval.llm_client import LLMClient, LLMMessage, LLMResponse, ToolCall
from smolval.mcp_client import MCPClientManager

logger = logging.getLogger(__name__)


class AgentStep(BaseModel):
    """Represents a single step in the agent's reasoning process."""

    iteration: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    tool_call_failed: bool = False
    llm_response: dict[str, Any] | None = None


class AgentResult(BaseModel):
    """Result from running the agent on a task."""

    success: bool
    final_answer: str
    steps: list[AgentStep]
    total_iterations: int
    error: str | None = None
    execution_time_seconds: float = 0.0
    token_usage: dict[str, int] | None = None
    failed_tool_calls: int = 0
    llm_responses: list[dict[str, Any]] = []


class Agent:
    """ReAct agent for evaluating MCP servers."""

    def __init__(
        self, config: Config, llm_client: LLMClient, mcp_manager: MCPClientManager
    ) -> None:
        """Initialize the agent."""
        self.config = config
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.conversation_history: list[LLMMessage] = []
        self._recent_steps_cache: list[AgentStep] = (
            []
        )  # Track recent steps for loop prevention

    async def run(
        self, prompt: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> AgentResult:
        """Run the agent on a given prompt using ReAct pattern."""
        try:
            # Apply timeout if configured
            if self.config.evaluation.timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._run_agent_loop(prompt, progress_callback),
                    timeout=self.config.evaluation.timeout_seconds,
                )
            else:
                return await self._run_agent_loop(prompt, progress_callback)
        except TimeoutError:
            logger.error(
                "Agent execution timed out after %d seconds",
                self.config.evaluation.timeout_seconds,
            )
            return AgentResult(
                success=False,
                final_answer="",
                steps=[],
                total_iterations=0,
                error=f"Agent execution timed out after {self.config.evaluation.timeout_seconds} seconds",
                execution_time_seconds=self.config.evaluation.timeout_seconds,
            )

    async def _run_agent_loop(
        self, prompt: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> AgentResult:
        """Internal method that runs the actual agent loop."""
        start_time = time.time()
        steps: list[AgentStep] = []
        # Reset the recent steps cache for this run
        self._recent_steps_cache = []
        total_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        failed_tool_calls = 0
        llm_responses: list[dict[str, Any]] = []

        try:
            # Initialize conversation with system prompt and user prompt
            self._reset_conversation()
            self._add_system_prompt()
            self._add_message("user", prompt)

            iteration = 0
            max_iterations = self.config.evaluation.max_iterations

            while iteration < max_iterations:
                iteration += 1

                # Update progress
                if progress_callback:
                    progress_callback(iteration, max_iterations)

                try:
                    # Get LLM response
                    logger.debug("Getting LLM response for iteration %d", iteration)
                    response = await self._get_llm_response()
                    logger.debug(
                        "Got response with %d tool calls",
                        len(response.tool_calls) if response.tool_calls else 0,
                    )

                    # Add assistant's response to conversation history
                    self._add_assistant_response(response)

                    # Track token usage
                    if response.token_usage:
                        total_token_usage["input_tokens"] += response.token_usage.get(
                            "input_tokens", 0
                        )
                        total_token_usage["output_tokens"] += response.token_usage.get(
                            "output_tokens", 0
                        )
                        total_token_usage["total_tokens"] += response.token_usage.get(
                            "total_tokens", 0
                        )

                    # Collect raw LLM response for debugging
                    if response.raw_response:
                        llm_responses.append(
                            {
                                "iteration": iteration,
                                "timestamp": time.time(),
                                "raw_response": response.raw_response,
                                "token_usage": response.token_usage,
                                "tool_calls_count": (
                                    len(response.tool_calls)
                                    if response.tool_calls
                                    else 0
                                ),
                            }
                        )

                    # Parse thought from response
                    thought = response.content or ""

                    # Check if LLM wants to use tools
                    if response.tool_calls:
                        # Execute tool calls
                        for tool_call in response.tool_calls:
                            try:
                                step, tool_failed = await self._execute_tool_call(
                                    iteration, thought, tool_call, response
                                )
                                if tool_failed:
                                    failed_tool_calls += 1
                                steps.append(step)
                                # Update recent steps cache for loop detection
                                self._recent_steps_cache.append(step)
                                if (
                                    len(self._recent_steps_cache) > 10
                                ):  # Keep only recent 10 steps
                                    self._recent_steps_cache.pop(0)

                                # Add tool result to conversation
                                tool_name, _, tool_id = self._extract_tool_call_info(
                                    tool_call
                                )
                                self._add_tool_result(tool_id, step.observation or "")
                            except Exception as tool_error:
                                # Log as debug instead of warning since tool failures are often expected
                                logger.debug("Tool execution failed: %s", tool_error)
                                failed_tool_calls += 1
                                tool_name, action_input, tool_id = (
                                    self._extract_tool_call_info(tool_call)
                                )

                                error_step = AgentStep(
                                    iteration=iteration,
                                    thought=thought,
                                    action=tool_name,
                                    action_input=action_input,
                                    observation=f"Error executing tool: {str(tool_error)}",
                                    tool_call_failed=True,
                                    llm_response={
                                        "content": response.content,
                                        "tool_calls": (
                                            [
                                                tc.model_dump()
                                                for tc in response.tool_calls
                                            ]
                                            if response.tool_calls
                                            else []
                                        ),
                                        "token_usage": response.token_usage,
                                        "raw_response": response.raw_response,
                                    },
                                )
                                steps.append(error_step)
                                # Update recent steps cache for loop detection
                                self._recent_steps_cache.append(error_step)
                                if (
                                    len(self._recent_steps_cache) > 10
                                ):  # Keep only recent 10 steps
                                    self._recent_steps_cache.pop(0)
                                self._add_tool_result(
                                    tool_id, error_step.observation or ""
                                )
                    else:
                        # No tool calls - this is the final answer
                        step = AgentStep(
                            iteration=iteration,
                            thought=thought,
                            action=None,
                            action_input=None,
                            observation=None,
                            llm_response={
                                "content": response.content,
                                "tool_calls": [],
                                "token_usage": response.token_usage,
                                "raw_response": response.raw_response,
                            },
                        )
                        steps.append(step)

                        execution_time = time.time() - start_time
                        return AgentResult(
                            success=True,
                            final_answer=thought,
                            steps=steps,
                            total_iterations=iteration,
                            execution_time_seconds=execution_time,
                            token_usage=(
                                total_token_usage
                                if total_token_usage["total_tokens"] > 0
                                else None
                            ),
                            failed_tool_calls=failed_tool_calls,
                            llm_responses=llm_responses,
                        )

                except Exception as e:
                    # Handle LLM errors - attempt recovery for recoverable errors
                    error_str = str(e)
                    logger.error("LLM error in iteration %d: %s", iteration, e)

                    # Check if this is a recoverable error (token limit)
                    # Also ensure we haven't already attempted recovery recently
                    recent_recovery_attempts = sum(
                        1
                        for step in steps[-3:]  # Check last 3 steps
                        if step.thought and "Token limit reached" in step.thought
                    )

                    if (
                        (
                            "prompt is too long" in error_str.lower()
                            or "maximum" in error_str.lower()
                            or "token" in error_str.lower()
                        )
                        and iteration
                        < max_iterations - 2  # Leave at least 2 iterations for recovery
                        and recent_recovery_attempts
                        < 2  # Prevent infinite recovery loops
                    ):
                        logger.info(
                            "Attempting error recovery for token limit issue - truncating conversation"
                        )

                        # Use sophisticated memory management to reduce token count
                        # Based on ConversationSummaryBufferMemory pattern
                        original_length = len(self.conversation_history)
                        if len(self.conversation_history) > 3:
                            # Always preserve system message and original task
                            system_and_task = []
                            for msg in self.conversation_history[:2]:
                                if msg.role in ["system", "user"]:
                                    system_and_task.append(msg)

                            # Keep the most recent 2-3 messages for immediate context
                            recent_messages = self.conversation_history[-2:]

                            # Summarize the middle conversation with key findings
                            middle_messages = self.conversation_history[
                                len(system_and_task) : -2
                            ]
                            if middle_messages:
                                # Extract key information from steps for summary
                                key_findings = []
                                successful_actions = []

                                for step in steps:
                                    if step.action and step.observation:
                                        if "error" not in step.observation.lower():
                                            successful_actions.append(
                                                f"{step.action}: {step.observation[:100]}..."
                                            )
                                        if any(
                                            keyword in step.observation.lower()
                                            for keyword in [
                                                "found",
                                                "discovered",
                                                "shows",
                                                "contains",
                                            ]
                                        ):
                                            key_findings.append(
                                                step.observation[:150] + "..."
                                            )

                                # Create intelligent summary
                                summary_parts = []
                                if successful_actions:
                                    summary_parts.append(
                                        f"Successfully completed actions: {'; '.join(successful_actions[:3])}"
                                    )
                                if key_findings:
                                    summary_parts.append(
                                        f"Key findings: {'; '.join(key_findings[:2])}"
                                    )
                                if len(steps) > 0:
                                    summary_parts.append(
                                        f"Completed {len(steps)} steps total"
                                    )

                                summary_content = (
                                    "[Conversation summarized] "
                                    + " | ".join(summary_parts)
                                )

                                summary_message = LLMMessage(
                                    role="assistant",
                                    content=summary_content,
                                )

                                # Reconstruct conversation: system+task + summary + recent context
                                self.conversation_history = (
                                    system_and_task
                                    + [summary_message]
                                    + recent_messages
                                )
                            else:
                                # Fallback: just keep system, task, and recent
                                self.conversation_history = (
                                    system_and_task + recent_messages
                                )

                            logger.info(
                                "Smart truncation: %d messages → %d messages, preserved key context",
                                original_length,
                                len(self.conversation_history),
                            )

                        # Add a recovery step with context preservation info
                        recovery_step = AgentStep(
                            iteration=iteration,
                            thought="Token limit reached - applied smart memory management",
                            action="",
                            observation="Context length exceeded limits. Applied intelligent summarization to preserve key findings while reducing token count. Continuing with condensed context.",
                            llm_response={
                                "content": "",
                                "tool_calls": [],
                                "token_usage": {},
                                "raw_response": {},
                            },
                        )
                        steps.append(recovery_step)
                        # Update recent steps cache for loop detection
                        self._recent_steps_cache.append(recovery_step)
                        if (
                            len(self._recent_steps_cache) > 10
                        ):  # Keep only recent 10 steps
                            self._recent_steps_cache.pop(0)

                        # Continue to next iteration with reduced context
                        continue

                    # Non-recoverable error - terminate
                    execution_time = time.time() - start_time
                    return AgentResult(
                        success=False,
                        final_answer="",
                        steps=steps,
                        total_iterations=iteration - 1,  # Don't count failed iteration
                        error=str(e),
                        execution_time_seconds=execution_time,
                        token_usage=(
                            total_token_usage
                            if total_token_usage["total_tokens"] > 0
                            else None
                        ),
                        failed_tool_calls=failed_tool_calls,
                        llm_responses=llm_responses,
                    )

            # Max iterations reached
            logger.warning("Maximum iterations (%d) exceeded", max_iterations)
            execution_time = time.time() - start_time
            return AgentResult(
                success=False,
                final_answer="",
                steps=steps,
                total_iterations=iteration,
                error=f"Maximum iterations ({max_iterations}) exceeded",
                execution_time_seconds=execution_time,
                token_usage=(
                    total_token_usage if total_token_usage["total_tokens"] > 0 else None
                ),
                failed_tool_calls=failed_tool_calls,
                llm_responses=llm_responses,
            )

        except Exception as e:
            # Top-level error (initialization, etc.)
            logger.error("Top-level agent error: %s", e)
            execution_time = time.time() - start_time
            return AgentResult(
                success=False,
                final_answer="",
                steps=steps,
                total_iterations=0,
                error=str(e),
                execution_time_seconds=execution_time,
                token_usage=(
                    total_token_usage if total_token_usage["total_tokens"] > 0 else None
                ),
                failed_tool_calls=failed_tool_calls,
                llm_responses=llm_responses,
            )

    async def _get_llm_response(self) -> LLMResponse:
        """Get response from LLM with available tools."""
        available_tools = self.mcp_manager.get_available_tools()

        # Call LLM with conversation history and available tools
        return await self.llm_client.chat(
            messages=self.conversation_history,
            tools=available_tools if available_tools else None,
        )

    def _extract_tool_call_info(
        self, tool_call: ToolCall | dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        """Extract tool call information from various formats."""
        if isinstance(tool_call, ToolCall):
            return (
                tool_call.name,
                tool_call.arguments,
                getattr(tool_call, "id", "unknown"),
            )
        elif isinstance(tool_call, dict):
            return (
                tool_call.get("name", "unknown"),
                tool_call.get("arguments", {}),
                tool_call.get("id", "unknown"),
            )
        else:
            raise ValueError(f"Unsupported tool call format: {type(tool_call)}")

    def _count_conversation_tokens(self) -> int:
        """Count current conversation token count using tiktoken."""
        try:
            # Use tiktoken for accurate token counting
            # Default to cl100k_base which is used by GPT-4, GPT-3.5-turbo, etc.
            encoding = tiktoken.get_encoding("cl100k_base")

            total_tokens = 0
            for msg in self.conversation_history:
                # Count tokens for role and content
                if msg.role:
                    total_tokens += len(encoding.encode(msg.role))
                if msg.content:
                    total_tokens += len(encoding.encode(msg.content))

            return total_tokens
        except Exception as e:
            logger.warning(
                "Failed to count tokens with tiktoken: %s. Using fallback estimation.",
                e,
            )
            # Fallback to character-based estimation
            total_chars = sum(
                len(msg.content or "") for msg in self.conversation_history
            )
            return total_chars // 3

    def _estimate_tool_output_tokens(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> int:
        """Estimate potential tool output size based on tool type and arguments."""
        # Tool-specific heuristics for size estimation
        size_estimates = {
            # File operations - can be very large
            "read_file": lambda args: (
                50000 if not args.get("max_lines") else args.get("max_lines", 1000) * 20
            ),
            "directory_tree": lambda args: 10000,  # Can be large for deep directories
            "list_directory": lambda args: 5000,  # Usually moderate size
            # Database/query operations - potentially massive
            "query": lambda args: 100000,  # Database queries can return huge results
            "get_resources": lambda args: 20000,  # Resource listings can be large
            # Web fetching - highly variable
            "fetch": lambda args: 30000,  # Web pages can be large
            # Default for unknown tools
            "default": lambda args: 10000,
        }

        estimator = size_estimates.get(tool_name, size_estimates["default"])
        return estimator(arguments)

    def _generate_tool_refinement_guidance(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Generate specific guidance for refining tool calls that would exceed token limits."""
        guidance_templates = {
            "read_file": (
                "The file you're trying to read would likely produce too much output for the context window. "
                "Try one of these approaches:\n"
                "1. Use 'read_file' with specific line ranges (e.g., lines 1-100)\n"
                "2. Look for specific content using grep-like search patterns\n"
                "3. Read just the beginning/end of the file to understand its structure\n"
                f"Current request: {arguments}"
            ),
            "directory_tree": (
                "The directory tree would be too large for the context window. "
                "Try these alternatives:\n"
                "1. Use 'list_directory' to explore one level at a time\n"
                "2. Focus on specific subdirectories that are relevant to your task\n"
                "3. Look for specific file types or patterns\n"
                f"Current request: {arguments}"
            ),
            "query": (
                "The database query would likely return too much data for the context window. "
                "Please refine your query:\n"
                "1. Add LIMIT clauses to restrict the number of rows\n"
                "2. Use more specific WHERE conditions to filter results\n"
                "3. Select only the columns you need instead of SELECT *\n"
                "4. Consider aggregating data (COUNT, SUM, AVG) instead of raw rows\n"
                f"Current query: {arguments}"
            ),
            "fetch": (
                "The web page you're trying to fetch might be too large for the context window. "
                "Consider:\n"
                "1. Looking for specific sections or elements of the page\n"
                "2. Searching for particular content instead of fetching the entire page\n"
                "3. Using more targeted URLs or API endpoints if available\n"
                f"Current request: {arguments}"
            ),
            "get_resources": (
                "The resource listing would be too extensive for the context window. "
                "Try:\n"
                "1. Filtering resources by type or category\n"
                "2. Looking for specific resource names or patterns\n"
                "3. Getting a summary or count instead of full details\n"
                f"Current request: {arguments}"
            ),
        }

        # Get specific guidance or provide generic guidance
        specific_guidance = guidance_templates.get(tool_name)
        if specific_guidance:
            return specific_guidance

        # Generic guidance for unknown tools
        return (
            f"The tool '{tool_name}' with the current parameters would likely produce too much output "
            f"for the context window. Please try:\n"
            f"1. Adding filters or limits to reduce the output size\n"
            f"2. Being more specific in your parameters\n"
            f"3. Breaking the request into smaller, more focused queries\n"
            f"Current parameters: {arguments}"
        )

    async def _execute_tool_call(
        self,
        iteration: int,
        thought: str,
        tool_call: ToolCall | dict[str, Any],
        llm_response: LLMResponse,
    ) -> tuple[AgentStep, bool]:
        """Execute a single tool call and return the step."""
        tool_name, arguments, tool_id = self._extract_tool_call_info(tool_call)

        try:
            # Check if tool execution would likely exceed token limits
            current_tokens = self._count_conversation_tokens()
            estimated_output_tokens = self._estimate_tool_output_tokens(
                tool_name, arguments
            )
            # Use configurable token budget for testing, default to 180k
            token_budget = getattr(self, "_test_token_budget", 180000)

            # Check for recent guidance to prevent guidance loops
            recent_guidance_count = 0
            if hasattr(self, "_recent_steps_cache"):
                recent_guidance_count = sum(
                    1
                    for step in self._recent_steps_cache[-3:]
                    if step.observation
                    and "would likely exceed token limit" in step.observation
                )

            if current_tokens + estimated_output_tokens > token_budget:
                if recent_guidance_count >= 2:
                    # Too many guidance attempts - force execution to avoid infinite loops
                    logger.warning(
                        "Too many consecutive guidance attempts (%d). Forcing tool execution to break potential loop.",
                        recent_guidance_count,
                    )
                    # Continue with normal execution below
                else:
                    logger.info(
                        "Tool %s with args %s would likely exceed token limit (%d + %d > %d). Providing guidance.",
                        tool_name,
                        arguments,
                        current_tokens,
                        estimated_output_tokens,
                        token_budget,
                    )

                    # Don't execute - provide guidance instead
                    guidance_message = self._generate_tool_refinement_guidance(
                        tool_name, arguments
                    )

                    step = AgentStep(
                        iteration=iteration,
                        thought=thought,
                        action=tool_name,
                        action_input=arguments,
                        observation=guidance_message,
                        llm_response={
                            "content": llm_response.content,
                            "tool_calls": (
                                [tc.model_dump() for tc in llm_response.tool_calls]
                                if llm_response.tool_calls
                                else []
                            ),
                            "token_usage": llm_response.token_usage,
                            "raw_response": llm_response.raw_response,
                        },
                    )

                    return step, False  # Not a "failure" - just guidance

            # Execute the tool normally
            logger.debug("Executing tool %s with arguments: %s", tool_name, arguments)
            result = await self.mcp_manager.execute_tool(tool_name, arguments)

            # Format observation and determine if failed
            failed = bool(result.error)
            if result.error:
                # Log as debug instead of warning since tool failures are often expected in evaluations
                logger.debug("Tool %s failed: %s", tool_name, result.error)
                observation = f"Tool execution failed: {result.error}"
            else:
                observation = result.content

                # Check if tool returned a JSON response indicating failure
                if observation:
                    try:
                        import json

                        parsed_response = json.loads(observation.strip())
                        if (
                            isinstance(parsed_response, dict)
                            and parsed_response.get("success") is False
                        ):
                            failed = True
                            logger.debug("Tool %s returned success:false", tool_name)
                    except (json.JSONDecodeError, AttributeError):
                        # Not JSON or parsing failed - that's fine, treat as success
                        pass

                if not failed:
                    logger.debug("Tool %s executed successfully", tool_name)

            step = AgentStep(
                iteration=iteration,
                thought=thought,
                action=tool_name,
                action_input=arguments,
                observation=observation,
                tool_call_failed=failed,
                llm_response={
                    "content": llm_response.content,
                    "tool_calls": (
                        [tc.model_dump() for tc in llm_response.tool_calls]
                        if llm_response.tool_calls
                        else []
                    ),
                    "token_usage": llm_response.token_usage,
                    "raw_response": llm_response.raw_response,
                },
            )

            return step, failed

        except Exception as e:
            logger.error("Unexpected error executing tool %s: %s", tool_name, e)
            step = AgentStep(
                iteration=iteration,
                thought=thought,
                action=tool_name,
                action_input=arguments,
                observation=f"Unexpected error executing tool: {str(e)}",
                tool_call_failed=True,
                llm_response={
                    "content": llm_response.content,
                    "tool_calls": (
                        [tc.model_dump() for tc in llm_response.tool_calls]
                        if llm_response.tool_calls
                        else []
                    ),
                    "token_usage": llm_response.token_usage,
                    "raw_response": llm_response.raw_response,
                },
            )
            return step, True  # Exception counts as failure

    def _reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history.clear()

    def _add_system_prompt(self) -> None:
        """Add system prompt for ReAct pattern."""
        system_prompt = (
            "You are a helpful assistant that can use tools to accomplish tasks. "
            "\n\nWhen you need to use tools, they will be provided to you as function calls. "
            "Use them when they can help accomplish the task.\n\n"
            "Think step by step about what you need to do and use the available tools as needed.\n\n"
            "When you have enough information to provide a final answer, provide your response directly."
        )

        self._add_message("system", system_prompt)

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(LLMMessage(role=role, content=content))

    def _add_assistant_response(self, response: LLMResponse) -> None:
        """Add assistant's response to conversation history."""
        self.conversation_history.append(
            LLMMessage(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
            )
        )

    def _add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Add tool result to conversation history."""
        self.conversation_history.append(
            LLMMessage(role="tool", content=result, tool_call_id=tool_call_id)
        )


class ClaudeCodeAgent:
    """Claude Code agent for evaluating MCP servers via CLI subprocess."""

    def __init__(self, config: Config, mcp_manager: MCPClientManager) -> None:
        """Initialize the Claude Code agent."""
        self.config = config
        self.mcp_manager = mcp_manager

    async def run(
        self, prompt: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> AgentResult:
        """Run the agent on a given prompt using Claude Code CLI."""
        start_time = time.time()
        
        try:
            # Apply timeout if configured
            if self.config.evaluation.timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._run_claude_code(prompt, progress_callback),
                    timeout=self.config.evaluation.timeout_seconds,
                )
            else:
                return await self._run_claude_code(prompt, progress_callback)
        except TimeoutError:
            logger.error(
                "Claude Code execution timed out after %d seconds",
                self.config.evaluation.timeout_seconds,
            )
            return AgentResult(
                success=False,
                final_answer="",
                steps=[],
                total_iterations=0,
                error=f"Claude Code execution timed out after {self.config.evaluation.timeout_seconds} seconds",
                execution_time_seconds=self.config.evaluation.timeout_seconds,
            )

    async def _run_claude_code(
        self, prompt: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> AgentResult:
        """Internal method that runs Claude Code via subprocess."""
        import json
        import tempfile
        import subprocess
        import shutil
        from pathlib import Path

        start_time = time.time()
        steps: list[AgentStep] = []
        
        try:
            # Set up MCP servers for Claude Code
            await self._setup_mcp_servers()
            
            # Progress callback
            if progress_callback:
                progress_callback(1, 2)

            # Find Claude executable
            claude_exe = self._find_claude_executable()
            if not claude_exe:
                raise RuntimeError("Claude Code CLI not found. Please install Claude Code CLI.")

            # Get tool permissions
            allowed_tools, disallowed_tools = self._get_tool_permissions()

            # Build Claude Code command
            cmd = [
                claude_exe,
                "-p", prompt,
                "--output-format", "stream-json",
                "--verbose"
            ]
            
            # Add allowed tools
            for tool in allowed_tools:
                cmd.extend(["--allowedTools", tool])
            
            # Add disallowed tools
            for tool in disallowed_tools:
                cmd.extend(["--disallowedTools", tool])

            # Add API key if available
            env = dict(os.environ)
            if self.config.llm.api_key:
                env["ANTHROPIC_API_KEY"] = self.config.llm.api_key

            logger.debug("Running Claude Code command: %s", " ".join(cmd))
            
            # Execute Claude Code subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Read and parse streaming JSON output
            stdout_data = b""
            stderr_data = b""
            
            if process.stdout:
                stdout_data, stderr_data = await process.communicate()
            
            return_code = process.returncode
            
            # Progress callback
            if progress_callback:
                progress_callback(2, 2)

            execution_time = time.time() - start_time
            
            if return_code != 0:
                error_msg = stderr_data.decode('utf-8') if stderr_data else "Claude Code failed"
                logger.error("Claude Code failed with return code %d: %s", return_code, error_msg)
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=[],
                    total_iterations=0,
                    error=error_msg,
                    execution_time_seconds=execution_time,
                )

            # Parse the streaming JSON output
            output_text = stdout_data.decode('utf-8') if stdout_data else ""
            final_answer, parsed_steps = self._parse_claude_code_output(output_text)
            
            return AgentResult(
                success=True,
                final_answer=final_answer,
                steps=parsed_steps,
                total_iterations=len(parsed_steps),
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Claude Code agent error: %s", e)
            return AgentResult(
                success=False,
                final_answer="",
                steps=[],
                total_iterations=0,
                error=str(e),
                execution_time_seconds=execution_time,
            )
        finally:
            # Clean up temporary files
            await self._cleanup_mcp_servers()

    async def _setup_mcp_servers(self) -> None:
        """Set up MCP servers for Claude Code using claude mcp commands."""
        import subprocess
        
        try:
            # Find Claude executable
            claude_exe = self._find_claude_executable()
            if not claude_exe:
                logger.warning("Claude Code CLI not found, skipping MCP server setup")
                return
            
            # Get existing MCP servers to avoid conflicts
            existing_servers = await self._get_existing_mcp_servers()
            
            for server_config in self.config.mcp_servers:
                server_name = f"smolval_{server_config.name}"
                
                # Skip if server already exists
                if server_name in existing_servers:
                    logger.debug("MCP server %s already exists, skipping", server_name)
                    continue
                
                # Build claude mcp add command
                cmd = [claude_exe, "mcp", "add", server_name]
                
                # Add environment variables
                for key, value in server_config.env.items():
                    cmd.extend(["-e", f"{key}={value}"])
                
                # Add server command
                cmd.append("--")
                cmd.extend(server_config.command)
                
                logger.debug("Adding MCP server: %s", " ".join(cmd))
                
                # Execute command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                _, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8') if stderr else "Failed to add MCP server"
                    logger.warning("Failed to add MCP server %s: %s", server_name, error_msg)
                
        except Exception as e:
            logger.warning("Error setting up MCP servers: %s", e)

    async def _get_existing_mcp_servers(self) -> set[str]:
        """Get list of existing MCP servers."""
        try:
            claude_exe = self._find_claude_executable()
            if not claude_exe:
                return set()
                
            process = await asyncio.create_subprocess_exec(
                claude_exe, "mcp", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                # Parse the output to extract server names
                output = stdout.decode('utf-8')
                # This is a simple parsing - may need adjustment based on actual output format
                servers = set()
                for line in output.splitlines():
                    if line.strip() and not line.startswith('#'):
                        # Extract server name (assuming first word is the name)
                        parts = line.strip().split()
                        if parts:
                            servers.add(parts[0])
                return servers
            
        except Exception as e:
            logger.debug("Error getting existing MCP servers: %s", e)
        
        return set()

    async def _cleanup_mcp_servers(self) -> None:
        """Clean up MCP servers added for this evaluation."""
        try:
            claude_exe = self._find_claude_executable()
            if not claude_exe:
                logger.debug("Claude Code CLI not found, skipping MCP server cleanup")
                return
                
            for server_config in self.config.mcp_servers:
                server_name = f"smolval_{server_config.name}"
                
                cmd = [claude_exe, "mcp", "remove", server_name]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if process.returncode != 0:
                    logger.debug("Failed to remove MCP server %s (may not have existed)", server_name)
                
        except Exception as e:
            logger.debug("Error cleaning up MCP servers: %s", e)

    def _parse_claude_code_output(self, output: str) -> tuple[str, list[AgentStep]]:
        """Parse Claude Code streaming JSON output into AgentResult format."""
        import json
        
        steps: list[AgentStep] = []
        final_answer = ""
        tool_uses = {}  # Track tool uses by ID
        
        # Split output into lines and parse each JSON object
        lines = output.strip().split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Extract different types of messages from Claude Code
                if isinstance(data, dict):
                    message_type = data.get('type', '')
                    
                    if message_type == 'result':
                        # This is the final result - use this as the clean final answer
                        final_answer = data.get('result', '')
                        break  # We found the final answer, stop processing
                    
                    elif message_type == 'assistant':
                        # Assistant message with tool uses or text
                        message = data.get('message', {})
                        content = message.get('content', [])
                        
                        for content_item in content:
                            if isinstance(content_item, dict):
                                if content_item.get('type') == 'tool_use':
                                    # Track tool use
                                    tool_id = content_item.get('id')
                                    tool_name = content_item.get('name')
                                    tool_input = content_item.get('input', {})
                                    
                                    tool_uses[tool_id] = {
                                        'name': tool_name,
                                        'input': tool_input,
                                        'step_index': len(steps)
                                    }
                                    
                                    step = AgentStep(
                                        iteration=len(steps) + 1,
                                        thought=f"Using tool: {tool_name}",
                                        action=tool_name,
                                        action_input=tool_input
                                    )
                                    steps.append(step)
                                
                                elif content_item.get('type') == 'text':
                                    # Text content
                                    text_content = content_item.get('text', '')
                                    if text_content:
                                        step = AgentStep(
                                            iteration=len(steps) + 1,
                                            thought=text_content,
                                            observation=None
                                        )
                                        steps.append(step)
                    
                    elif message_type == 'user':
                        # User message with tool results
                        message = data.get('message', {})
                        content = message.get('content', [])
                        
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get('type') == 'tool_result':
                                tool_id = content_item.get('tool_use_id')
                                result_content = content_item.get('content', '')
                                
                                # Find the corresponding tool use and add the result
                                if tool_id in tool_uses:
                                    step_index = tool_uses[tool_id]['step_index']
                                    if step_index < len(steps):
                                        steps[step_index].observation = result_content
                                        
            except json.JSONDecodeError:
                # If not valid JSON, treat as plain text
                if line.strip():
                    step = AgentStep(
                        iteration=len(steps) + 1,
                        thought=line.strip(),
                        observation=None
                    )
                    steps.append(step)
        
        # If no final answer was found, create one from the steps
        if not final_answer:
            if steps:
                # Use the last step's thought as the final answer
                final_answer = steps[-1].thought
            else:
                # Fallback to raw output
                final_answer = output.strip()
        
        # If no steps were created, create a single step
        if not steps:
            steps.append(AgentStep(
                iteration=1,
                thought=final_answer,
                observation=None
            ))
        
        return final_answer.strip(), steps

    def _find_claude_executable(self) -> str | None:
        """Find the Claude Code CLI executable."""
        import shutil
        from pathlib import Path
        
        # First try to find claude in PATH
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path
        
        # Common installation paths
        possible_paths = [
            Path.home() / ".claude" / "local" / "claude",
            Path("/usr/local/bin/claude"),
            Path("/opt/homebrew/bin/claude"),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                return str(path)
        
        return None

    def _get_tool_permissions(self) -> tuple[list[str], list[str]]:
        """Get tool permissions for Claude Code CLI flags."""
        
        # Define allowed tools for MCP evaluation tasks (including development)
        allowed_tools = [
            # File system operations (full access for development)
            "LS(**)",
            "Read(**)",
            "Edit(**)",
            "Write(**)",
            "Glob(**)",
            "Grep(**)",
            "MultiEdit(**)",
            
            # Task management and automation
            "Task(**)",
            "TodoRead(**)",
            "TodoWrite(**)",
            
            # Web operations (for fetch MCP server)
            "WebFetch(**)",
            "WebSearch(**)",
            
            # Bash operations (development and file management)
            "Bash(ls *)",
            "Bash(cat *)",
            "Bash(grep *)",
            "Bash(find *)",
            "Bash(pwd)",
            "Bash(cd *)",
            "Bash(head *)",
            "Bash(tail *)",
            "Bash(wc *)",
            "Bash(sort *)",
            "Bash(uniq *)",
            "Bash(cut *)",
            "Bash(awk *)",
            "Bash(sed *)",
            "Bash(echo *)",
            "Bash(which *)",
            "Bash(type *)",
            
            # Directory and file creation (needed for projects)
            "Bash(mkdir *)",
            "Bash(touch *)",
            "Bash(cp *.* *)",  # Copy files (not directories)
            "Bash(mv *.* *)",  # Move files (not directories)
            
            # Development tools and package managers (essential for SPA development)
            "Bash(npm init *)",
            "Bash(npm install *)",
            "Bash(npm run *)",
            "Bash(npm start)",
            "Bash(npm test)",
            "Bash(npm build)",
            "Bash(npx *)",
            "Bash(yarn *)",
            "Bash(node *)",
            "Bash(python *)",
            "Bash(pip install *)",
            
            # Git operations (project management)
            "Bash(git init)",
            "Bash(git status)",
            "Bash(git add *)",
            "Bash(git commit *)",
            "Bash(git log *)",
            "Bash(git diff *)",
            "Bash(git show *)",
            
            # Development servers and tools
            "Bash(python -m http.server *)",
            "Bash(python3 -m http.server *)",
            "Bash(live-server *)",
            "Bash(serve *)",
            
            # MCP GitHub tools (if available)
            "mcp__github__get_file_contents(**)",
            "mcp__github__get_issue(**)",
            "mcp__github__get_pull_request(**)",
            "mcp__github__list_issues(**)",
            "mcp__github__list_pull_requests(**)",
            "mcp__github__search_code(**)",
            "mcp__github__search_issues(**)",
            "mcp__github__search_repositories(**)",
        ]
        
        # Define disallowed tools (truly dangerous operations)
        disallowed_tools = [
            # File deletion and dangerous operations
            "Bash(rm *)",
            "Bash(rmdir *)",
            "Bash(rm -rf *)",
            "Bash(rm -f *)",
            
            # Permission and ownership changes
            "Bash(chmod *)",
            "Bash(chown *)",
            "Bash(chgrp *)",
            
            # Privilege escalation
            "Bash(sudo *)",
            "Bash(su *)",
            
            # Network operations (could be used maliciously)
            "Bash(curl *)",
            "Bash(wget *)",
            "Bash(ssh *)",
            "Bash(scp *)",
            "Bash(rsync *)",
            "Bash(nc *)",
            "Bash(netcat *)",
            
            # Process management
            "Bash(killall *)",
            "Bash(pkill *)",
            "Bash(kill *)",
            
            # System service management
            "Bash(systemctl *)",
            "Bash(service *)",
            "Bash(launchctl *)",
            
            # System package managers (global installs)
            "Bash(brew install *)",
            "Bash(brew uninstall *)",
            "Bash(apt *)",
            "Bash(yum *)",
            "Bash(dnf *)",
            "Bash(pacman *)",
            "Bash(zypper *)",
            
            # Global npm operations
            "Bash(npm install -g *)",
            "Bash(npm uninstall -g *)",
            "Bash(npm link *)",
            "Bash(npm unlink *)",
            
            # Directory operations on system paths
            "Bash(mv /* *)",
            "Bash(cp -r /* *)",
            "Bash(mkdir /usr *)",
            "Bash(mkdir /etc *)",
            "Bash(mkdir /var *)",
            "Bash(mkdir /sys *)",
            "Bash(mkdir /proc *)",
            
            # MCP GitHub write operations (read-only for evaluation)
            "mcp__github__create_issue(**)",
            "mcp__github__update_issue(**)",
            "mcp__github__create_pull_request(**)",
            "mcp__github__update_pull_request(**)",
            "mcp__github__merge_pull_request(**)",
            "mcp__github__create_or_update_file(**)",
            "mcp__github__delete_file(**)",
            "mcp__github__push_files(**)",
        ]
        
        logger.debug("Configured %d allowed tools and %d disallowed tools", 
                    len(allowed_tools), len(disallowed_tools))
        return allowed_tools, disallowed_tools
