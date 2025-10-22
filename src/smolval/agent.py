"""Claude Code agent implementation for MCP server evaluation."""

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    UserMessage,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    ClaudeSDKClient,
)
from smolval.models.agent import AgentStep, ExecutionMetadata


from .models import AgentResult


logger = logging.getLogger(__name__)


class ClaudeCodeAgent:
    """Claude Code agent for evaluating MCP servers - shared across all prompts."""

    def __init__(
        self,
        mcp_config_path: str | None = None,
        timeout_seconds: int = 900,
        verbose: bool = False,
        env_file: str | None = None,
        allowed_mcp_tools: str | None = None,
    ) -> None:
        """Initialize the Claude Code agent."""
        self.mcp_config_path = mcp_config_path or ".mcp.json"
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        self.env_file = env_file
        self.allowed_mcp_tools = allowed_mcp_tools

        # Load environment variables from .env file
        env_file_path = env_file or ".env"
        if os.path.exists(env_file_path):
            load_dotenv(env_file_path, override=True)
            logger.debug(f"Loaded environment from {env_file_path}")
        else:
            load_dotenv()  # Try default behavior
            logger.debug("Loaded environment using default .env discovery")

    def _build_allowed_tools_arg(self) -> list[str]:
        """Build the --allowed-tools argument with built-in tools and optional MCP tools."""
        # Built-in tools that require permissions
        builtin_tools = [
            "Bash",
            "Edit",
            "MultiEdit",
            "NotebookEdit",
            "WebFetch",
            "WebSearch",
            "Write",
        ]

        # Start with built-in tools
        allowed_tools = builtin_tools[:]

        # Add MCP tools if provided
        if self.allowed_mcp_tools:
            # Split by comma and clean up whitespace
            mcp_tools = [
                tool.strip()
                for tool in self.allowed_mcp_tools.split(",")
                if tool.strip()
            ]
            allowed_tools.extend(mcp_tools)

        return allowed_tools

    async def run(self, prompt: str, show_progress: bool = True) -> AgentResult:
        """Run the agent on a given prompt using Claude Code CLI."""
        import uuid
        from datetime import datetime

        from .models import ExecutionMetadata

        start_time = time.time()
        try:
            # Apply timeout if configured
            if self.timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._run_claude_code(prompt, show_progress),
                    timeout=self.timeout_seconds,
                )
            else:
                return await self._run_claude_code(prompt, show_progress)
        except TimeoutError:
            execution_time = time.time() - start_time
            logger.error(
                "Claude Code execution timed out after %d seconds",
                self.timeout_seconds,
            )

            metadata = ExecutionMetadata(
                session_id=str(uuid.uuid4()),
                model_used="unknown",
                execution_start=datetime.fromtimestamp(start_time),
                execution_end=datetime.now(),
            )

            return AgentResult(
                success=False,
                final_answer="",
                error_message=f"Claude Code execution timed out after {self.timeout_seconds} seconds",
                steps=[],
                total_iterations=0,
                execution_time_seconds=execution_time,
                metadata=metadata,
            )

    async def _run_claude_code(
        self, prompt: str, show_progress: bool = True
    ) -> AgentResult:
        """Internal method that runs Claude Code via SDK."""

        session = ClaudeCodeAgentSession(self, show_progress)
        return await session.run(prompt)

    def _build_environment(self) -> dict[str, str]:
        """Build the environment dictionary."""
        # Load additional environment variables from .env if present
        env_file_path = self.env_file or ".env"
        if os.path.exists(env_file_path):
            load_dotenv(env_file_path, override=True)
        else:
            load_dotenv()  # Try default behavior

        env = dict(os.environ)

        if "ANTHROPIC_API_KEY" not in env:
            logger.warning("ANTHROPIC_API_KEY not found in environment")

        return env

    def _find_mcp_path(self) -> Path | None:
        # Handle MCP config path - validate file exists and is valid JSON
        mcp_config_abs_path = os.path.abspath(self.mcp_config_path)

        if os.path.exists(mcp_config_abs_path):
            # Check if file is not empty and is valid JSON
            try:
                if os.path.getsize(mcp_config_abs_path) > 0:
                    import json

                    with open(mcp_config_abs_path) as f:
                        json.load(f)
                        return Path(mcp_config_abs_path)
                else:
                    logger.warning(
                        f"MCP config file {mcp_config_abs_path} is empty, using Claude Code's built-in tools only"
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"MCP config file {mcp_config_abs_path} is invalid JSON ({e}), using Claude Code's built-in tools only"
                )
        else:
            logger.debug(
                f"MCP config file not found at {self.mcp_config_path}, using Claude Code's built-in tools only"
            )
        return {}


class ClaudeCodeAgentSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, agent: ClaudeCodeAgent, show_progress: bool = True):
        self.agent = agent
        self.show_progress = show_progress
        mcp_path = agent._find_mcp_path()
        config_dir = os.getcwd()
        if mcp_path:
            config_dir = mcp_path.parent
        options = ClaudeAgentOptions(
            mcp_servers=mcp_path,
            permission_mode="bypassPermissions",
            allowed_tools=agent._build_allowed_tools_arg(),
            env=agent._build_environment(),
            cwd=config_dir,
        )
        self.client = ClaudeSDKClient(options=options)
        self.current_iteration = 0
        self.steps: list[AgentStep] = []
        self.start_time = datetime.datetime.now()

    def _add_step(self, message: Message) -> AgentResult | None:
        iteration = self.current_iteration
        if isinstance(message, SystemMessage):
            self.steps.append(
                AgentStep(
                    step_id=message.data["uuid"],
                    iteration=iteration,
                    step_type="system_init",
                    content=json.dumps(message.data, indent=4),
                    raw_claude_message=message,
                )
            )
        elif isinstance(message, AssistantMessage) or isinstance(message, UserMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    self.steps.append(
                        AgentStep(
                            step_id=str(uuid.uuid4()),
                            iteration=iteration,
                            step_type="text_response",
                            content=block.text,
                            model_used=message.model,
                            raw_claude_message=message,
                        )
                    )
                elif isinstance(block, ToolUseBlock):
                    self.steps.append(
                        AgentStep(
                            step_id=str(uuid.uuid4()),
                            iteration=iteration,
                            step_type="tool_use",
                            content=f"Using tool: {block.name}",
                            tool_call_id=block.id,
                            tool_name=block.name,
                            tool_input=block.input,
                            model_used=message.model,
                            raw_claude_message=message,
                        )
                    )
                elif isinstance(block, ToolResultBlock):
                    self.steps.append(
                        AgentStep(
                            step_id=str(uuid.uuid4()),
                            iteration=iteration,
                            step_type="tool_result",
                            content=block.content,
                            tool_call_id=block.tool_use_id,
                            tool_output=block.content,
                            tool_error=not not block.is_error,
                            raw_claude_message=message,
                        )
                    )
        elif isinstance(message, ResultMessage):
            return AgentResult(
                success=not message.is_error,
                final_answer=message.result,
                steps=self.steps,
                total_iterations=message.num_turns,
                execution_time_seconds=message.duration_ms / 1000,
                metadata=ExecutionMetadata(
                    session_id=message.session_id,
                    model_used="unknown",
                    execution_start=self.start_time,
                    execution_end=datetime.datetime.now(),
                ),
            )

    async def run(self, prompt: str) -> AgentResult:
        await self.client.connect()
        start_time = time.time()

        try:
            # Log initialization step
            logger.debug("Initializing Claude Code execution")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=False,
                # disable=not self.show_progress,
            ) as progress:
                progress.add_task("Claude Code executing...", total=None)
                await self.client.query(prompt)
                final_result = None

                async for msg in self.client.receive_response():
                    self.current_iteration += 1
                    if self.agent.verbose:
                        progress.console.print(msg)
                    maybe_result = self._add_step(msg)
                    if maybe_result:
                        # Claude cautions against returning early, so we wait until the end to return the result
                        final_result = maybe_result

            if final_result:
                return final_result

            return AgentResult(
                success=False,
                final_answer="",
                error_message="No result message received - Claude Code execution timed out",
                steps=[],
                total_iterations=0,
                execution_time_seconds=0.0,
                metadata=ExecutionMetadata(
                    session_id=str(uuid.uuid4()),
                    model_used="unknown",
                    execution_start=start_time,
                    execution_end=time.time(),
                ),
            )

        except Exception as e:
            logger.error("Claude Code agent error: %s", e)
            return AgentResult(
                success=False,
                final_answer="",
                error_message=str(e),
                steps=[],
                total_iterations=0,
                execution_time_seconds=0.0,
                metadata={
                    ExecutionMetadata(
                        session_id=str(uuid.uuid4()),
                        model_used="unknown",
                        execution_start=start_time,
                        execution_end=time.time(),
                    )
                },
            )
        finally:
            await self.client.disconnect()
