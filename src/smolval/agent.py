"""Claude Code agent implementation for MCP server evaluation."""

import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .claude_parser import ClaudeStreamParser
from .models import AgentResult

logger = logging.getLogger(__name__)


class ClaudeCodeAgent:
    """Claude Code agent for evaluating MCP servers via CLI subprocess."""

    def __init__(
        self,
        mcp_config_path: str | None = None,
        timeout_seconds: int = 300,
        verbose: bool = False,
        env_file: str | None = None,
        isolate_mcp_config: bool = False,
        allowed_mcp_tools: str | None = None,
    ) -> None:
        """Initialize the Claude Code agent."""
        self.mcp_config_path = mcp_config_path or ".mcp.json"
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        self.env_file = env_file
        self.isolate_mcp_config = isolate_mcp_config
        self.allowed_mcp_tools = allowed_mcp_tools

        # Load environment variables from .env file
        env_file_path = env_file or ".env"
        if os.path.exists(env_file_path):
            load_dotenv(env_file_path, override=True)
            logger.debug(f"Loaded environment from {env_file_path}")
        else:
            load_dotenv()  # Try default behavior
            logger.debug("Loaded environment using default .env discovery")

    def _build_allowed_tools_arg(self) -> str:
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

        # Join with commas for the CLI argument
        return ",".join(allowed_tools)

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
        """Internal method that runs Claude Code via subprocess."""
        start_time = time.time()

        try:
            # Find Claude executable
            claude_exe = self._find_claude_executable()
            if not claude_exe:
                raise RuntimeError(
                    "Claude Code CLI not found. Please install Claude Code CLI."
                )

            # Build Claude Code command
            cmd = [
                claude_exe,
                "-p",
                prompt,
                "--output-format",
                "stream-json",  # Use stream-json for real-time output
                "--verbose",  # Required for stream-json output
            ]

            # Add allowed tools for evaluation scenarios
            allowed_tools_arg = self._build_allowed_tools_arg()
            cmd.extend(["--allowed-tools", allowed_tools_arg])
            logger.debug(f"Added --allowed-tools flag: {allowed_tools_arg}")

            # Handle MCP config path - validate file exists and is valid JSON
            mcp_config_abs_path = os.path.abspath(self.mcp_config_path)
            config_dir = os.getcwd()

            if os.path.exists(mcp_config_abs_path):
                # Check if file is not empty and is valid JSON
                try:
                    if os.path.getsize(mcp_config_abs_path) > 0:
                        import json

                        with open(mcp_config_abs_path) as f:
                            json.load(f)  # Validate JSON

                        # File is valid, use it
                        config_dir = os.path.dirname(mcp_config_abs_path)
                        cmd.extend(["--mcp-config", mcp_config_abs_path])
                        logger.debug(
                            f"Added explicit MCP config: {mcp_config_abs_path}"
                        )
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

            # Set up environment - ensure all environment variables are passed
            env = dict(os.environ)

            # Load additional environment variables from .env if present
            # (we already loaded in __init__, but reload to ensure subprocess gets them)
            env_file_path = self.env_file or ".env"
            if os.path.exists(env_file_path):
                load_dotenv(env_file_path, override=True)
                env.update(os.environ)  # Update with newly loaded vars
                logger.debug(
                    f"Reloaded environment from {env_file_path} for subprocess"
                )
            else:
                load_dotenv()  # Try default behavior
                env.update(os.environ)

            if "ANTHROPIC_API_KEY" not in env:
                logger.warning("ANTHROPIC_API_KEY not found in environment")

            logger.debug("Running Claude Code command: %s", " ".join(cmd))
            logger.debug("Working directory: %s", config_dir)
            logger.debug(
                "Environment variables: %s",
                {k: v for k, v in env.items() if "API_KEY" not in k},
            )

            # Test Claude CLI first
            if self.verbose:
                print("Testing Claude CLI availability...", file=sys.stderr)
                try:
                    test_result = await asyncio.create_subprocess_exec(
                        claude_exe,
                        "--version",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                    )
                    test_stdout, test_stderr = await test_result.communicate()
                    print(
                        f"Claude CLI version check: exit_code={test_result.returncode}",
                        file=sys.stderr,
                    )
                    if test_stdout:
                        print(
                            f"Claude version: {test_stdout.decode().strip()}",
                            file=sys.stderr,
                        )
                    if test_stderr:
                        print(
                            f"Claude version stderr: {test_stderr.decode().strip()}",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(f"Claude CLI test failed: {e}", file=sys.stderr)

            # Log initialization step
            logger.debug("Initializing Claude Code execution")

            # Execute Claude Code subprocess with progress indicator
            if show_progress and not self.verbose:
                # Use Rich progress indicator for non-verbose mode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    transient=False,
                ) as progress:
                    progress.add_task("Claude Code executing...", total=None)

                    # Start subprocess
                    if claude_exe == "claude":
                        # Use shell escaping for safety
                        import shlex

                        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
                        process = await asyncio.create_subprocess_shell(
                            cmd_str,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                            cwd=config_dir,
                        )
                    else:
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                            cwd=config_dir,
                        )

                    # Capture stdout and stderr
                    stdout_data, stderr_data = await process.communicate()
                    return_code = process.returncode
            else:
                # No progress indicator for verbose mode or when disabled
                if claude_exe == "claude":
                    # Use shell escaping for safety
                    import shlex

                    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
                    process = await asyncio.create_subprocess_shell(
                        cmd_str,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=config_dir,
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=config_dir,
                    )

                # Capture stdout and stderr
                stdout_data, stderr_data = await process.communicate()
                return_code = process.returncode

            # Log output if verbose or for debugging
            stdout_text = (
                stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            )
            stderr_text = (
                stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            )

            if self.verbose and stdout_text:
                print(f"Claude stdout:\n{stdout_text}", file=sys.stderr)
            if self.verbose and stderr_text:
                print(f"Claude stderr:\n{stderr_text}", file=sys.stderr)

            if stdout_text:
                logger.debug("Claude stdout: %s", stdout_text)
            if stderr_text:
                logger.debug("Claude stderr: %s", stderr_text)

            # Log completion
            logger.debug("Claude Code execution completed")

            execution_time = time.time() - start_time

            if return_code != 0:
                error_msg = (
                    stderr_data.decode("utf-8") if stderr_data else "Claude Code failed"
                )
                logger.error(
                    "Claude Code failed with return code %d: %s", return_code, error_msg
                )
                import uuid
                from datetime import datetime

                from .models import ExecutionMetadata

                metadata = ExecutionMetadata(
                    session_id=str(uuid.uuid4()),
                    model_used="unknown",
                    execution_start=datetime.fromtimestamp(start_time),
                    execution_end=datetime.now(),
                )

                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=[],
                    total_iterations=0,
                    error_message=error_msg,
                    execution_time_seconds=execution_time,
                    metadata=metadata,
                )

            # Parse the streaming JSON output using the new parser
            parser = ClaudeStreamParser()
            return parser.parse_stream(stdout_text)

        except Exception as e:
            logger.error("Claude Code agent error: %s", e)
            parser = ClaudeStreamParser()
            return parser._create_error_result(str(e))

    def _find_claude_executable(self) -> str | None:
        """Find the Claude Code CLI executable."""
        import shutil
        import subprocess

        # Check if we're running in a container by looking for container-specific indicators
        in_container = (
            os.path.exists("/.dockerenv")
            or os.path.exists("/app/.claude")
            or os.environ.get("CLAUDE_CONFIG_DIR") == "/app/.claude"
        )

        # In container, use the pre-installed Claude CLI path first
        if in_container:
            container_claude_path = "/usr/local/bin/claude"
            if os.path.exists(container_claude_path) and os.access(
                container_claude_path, os.X_OK
            ):
                logger.debug(f"Using container Claude CLI: {container_claude_path}")
                return container_claude_path

        # First try to use shell to resolve aliases
        try:
            result = subprocess.run(
                ["which", "claude"], capture_output=True, text=True, shell=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        # Try checking if shell alias exists by running claude --version
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                shell=True,  # Use shell to resolve aliases
            )
            if result.returncode == 0:
                return "claude"  # Shell can resolve it
        except Exception:
            pass

        # Fall back to common locations (container path first, then host paths)
        claude_paths = [
            "/usr/local/bin/claude",  # Container and common install location
            "/opt/homebrew/bin/claude",  # macOS Homebrew
        ]

        for path in claude_paths:
            if shutil.which(path):
                return path

        # Try additional installation locations for Claude Code
        claude_paths.extend(
            [
                os.path.expanduser("~/.claude/local/claude"),
                os.path.expanduser("~/.local/bin/claude"),
            ]
        )

        for path in claude_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        return None
