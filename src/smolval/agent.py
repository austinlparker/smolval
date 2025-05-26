"""ReAct agent implementation for MCP server evaluation."""

import logging
import time
from typing import Any, Callable

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

    def __init__(self, config: Config, llm_client: LLMClient, mcp_manager: MCPClientManager) -> None:
        """Initialize the agent."""
        self.config = config
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.conversation_history: list[LLMMessage] = []

    async def run(self, prompt: str, progress_callback: Callable[[int, int], None] | None = None) -> AgentResult:
        """Run the agent on a given prompt using ReAct pattern."""
        start_time = time.time()
        steps: list[AgentStep] = []
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
                    logger.debug("Got response with %d tool calls", len(response.tool_calls) if response.tool_calls else 0)

                    # Add assistant's response to conversation history
                    self._add_assistant_response(response)

                    # Track token usage
                    if response.token_usage:
                        total_token_usage["input_tokens"] += response.token_usage.get("input_tokens", 0)
                        total_token_usage["output_tokens"] += response.token_usage.get("output_tokens", 0)
                        total_token_usage["total_tokens"] += response.token_usage.get("total_tokens", 0)

                    # Collect raw LLM response for debugging
                    if response.raw_response:
                        llm_responses.append({
                            "iteration": iteration,
                            "timestamp": time.time(),
                            "raw_response": response.raw_response,
                            "token_usage": response.token_usage,
                            "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0
                        })

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

                                # Add tool result to conversation
                                tool_name, _, tool_id = self._extract_tool_call_info(tool_call)
                                self._add_tool_result(tool_id, step.observation or "")
                            except Exception as tool_error:
                                logger.warning("Tool execution failed: %s", tool_error)
                                failed_tool_calls += 1
                                tool_name, action_input, tool_id = self._extract_tool_call_info(tool_call)

                                error_step = AgentStep(
                                    iteration=iteration,
                                    thought=thought,
                                    action=tool_name,
                                    action_input=action_input,
                                    observation=f"Error executing tool: {str(tool_error)}",
                                    llm_response={
                                        "content": response.content,
                                        "tool_calls": [tc.model_dump() for tc in response.tool_calls] if response.tool_calls else [],
                                        "token_usage": response.token_usage,
                                        "raw_response": response.raw_response
                                    }
                                )
                                steps.append(error_step)
                                self._add_tool_result(tool_id, error_step.observation or "")
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
                                "raw_response": response.raw_response
                            }
                        )
                        steps.append(step)

                        execution_time = time.time() - start_time
                        return AgentResult(
                            success=True,
                            final_answer=thought,
                            steps=steps,
                            total_iterations=iteration,
                            execution_time_seconds=execution_time,
                            token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                            failed_tool_calls=failed_tool_calls,
                            llm_responses=llm_responses
                        )

                except Exception as e:
                    # Handle LLM errors - this should terminate the agent
                    logger.error("LLM error in iteration %d: %s", iteration, e)
                    execution_time = time.time() - start_time
                    return AgentResult(
                        success=False,
                        final_answer="",
                        steps=steps,
                        total_iterations=iteration - 1,  # Don't count failed iteration
                        error=str(e),
                        execution_time_seconds=execution_time,
                        token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                        failed_tool_calls=failed_tool_calls,
                        llm_responses=llm_responses
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
                token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                failed_tool_calls=failed_tool_calls,
                llm_responses=llm_responses
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
                token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                failed_tool_calls=failed_tool_calls,
                llm_responses=llm_responses
            )

    async def _get_llm_response(self) -> LLMResponse:
        """Get response from LLM with available tools."""
        available_tools = self.mcp_manager.get_available_tools()

        # Call LLM with conversation history and available tools
        return await self.llm_client.chat(
            messages=self.conversation_history,
            tools=available_tools if available_tools else None
        )

    def _extract_tool_call_info(self, tool_call: ToolCall | dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """Extract tool call information from various formats."""
        if hasattr(tool_call, 'name'):
            return tool_call.name, tool_call.arguments, getattr(tool_call, 'id', 'unknown')
        elif isinstance(tool_call, dict):
            return tool_call.get('name', 'unknown'), tool_call.get('arguments', {}), tool_call.get('id', 'unknown')
        else:
            raise ValueError(f"Unsupported tool call format: {type(tool_call)}")

    async def _execute_tool_call(self, iteration: int, thought: str, tool_call: ToolCall | dict[str, Any], llm_response: LLMResponse) -> tuple[AgentStep, bool]:
        """Execute a single tool call and return the step."""
        tool_name, arguments, tool_id = self._extract_tool_call_info(tool_call)

        try:
            # Execute the tool
            logger.debug("Executing tool %s with arguments: %s", tool_name, arguments)
            result = await self.mcp_manager.execute_tool(tool_name, arguments)

            # Format observation and determine if failed
            failed = bool(result.error)
            if result.error:
                logger.warning("Tool %s failed: %s", tool_name, result.error)
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
                    "tool_calls": [tc.model_dump() for tc in llm_response.tool_calls] if llm_response.tool_calls else [],
                    "token_usage": llm_response.token_usage,
                    "raw_response": llm_response.raw_response
                }
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
                    "tool_calls": [tc.model_dump() for tc in llm_response.tool_calls] if llm_response.tool_calls else [],
                    "token_usage": llm_response.token_usage,
                    "raw_response": llm_response.raw_response
                }
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
        self.conversation_history.append(LLMMessage(
            role=role,
            content=content
        ))

    def _add_assistant_response(self, response: LLMResponse) -> None:
        """Add assistant's response to conversation history."""
        self.conversation_history.append(LLMMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        ))

    def _add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Add tool result to conversation history."""
        self.conversation_history.append(LLMMessage(
            role="tool",
            content=result,
            tool_call_id=tool_call_id
        ))
