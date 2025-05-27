"""ReAct agent implementation for MCP server evaluation."""

import logging
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
                logger.debug("Tool %s executed successfully", tool_name)
                observation = result.content

            step = AgentStep(
                iteration=iteration,
                thought=thought,
                action=tool_name,
                action_input=arguments,
                observation=observation,
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
