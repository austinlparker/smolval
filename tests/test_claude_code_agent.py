"""Tests for ClaudeCodeAgent."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from smolval.agent import ClaudeCodeAgent
from smolval.models import AgentResult


@pytest.fixture
def agent():
    """Create a Claude Code agent."""
    return ClaudeCodeAgent(mcp_config_path=".mcp.json", timeout_seconds=60)


class TestClaudeCodeAgent:
    """Test Claude Code agent functionality."""

    def test_init(self):
        """Test agent initialization."""
        agent = ClaudeCodeAgent(
            mcp_config_path="/path/to/.mcp.json", timeout_seconds=120
        )
        assert agent.mcp_config_path == "/path/to/.mcp.json"
        assert agent.timeout_seconds == 120

    def test_init_defaults(self):
        """Test agent initialization with defaults."""
        agent = ClaudeCodeAgent()
        assert agent.mcp_config_path == ".mcp.json"
        assert agent.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_claude_stream_parser_empty_output(self):
        """Test parsing empty output."""
        from smolval.claude_parser import ClaudeStreamParser

        parser = ClaudeStreamParser()
        result = parser.parse_stream("")

        assert not result.success
        assert result.error_message == "No output received from Claude Code"
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_claude_stream_parser_json(self):
        """Test parsing proper stream-json output."""
        from smolval.claude_parser import ClaudeStreamParser

        # Proper Claude Code stream-json output
        output = """{"type":"assistant","message":{"id":"msg_1","type":"message","content":[{"type":"text","text":"Processing task..."}],"model":"claude-sonnet-4"},"session_id":"test-123"}
{"type":"assistant","message":{"id":"msg_2","type":"message","content":[{"type":"tool_use","name":"Read","input":{"file_path":"test.txt"},"id":"tool_1"}],"model":"claude-sonnet-4"},"session_id":"test-123"}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tool_1","content":"File content here"}]},"session_id":"test-123"}
{"type":"result","subtype":"success","result":"Task completed successfully!","session_id":"test-123"}"""

        parser = ClaudeStreamParser()
        result = parser.parse_stream(output)

        assert result.success
        assert result.final_answer == "Task completed successfully!"
        assert (
            len(result.steps) >= 3
        )  # At least one text, one tool use, one tool result

        # Check that we have the expected step types
        step_types = [step.step_type for step in result.steps]
        assert "text_response" in step_types
        assert "tool_use" in step_types
        assert "tool_result" in step_types

    @pytest.mark.asyncio
    async def test_claude_stream_parser_list_content(self):
        """Test parsing tool result with list content."""
        from smolval.claude_parser import ClaudeStreamParser

        # Claude output with list content in tool result
        output = """{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tool_1","content":[{"type":"text","text":"Multiple content items here"},{"type":"text","text":" and more text."}]}]},"session_id":"test-123"}
{"type":"result","subtype":"success","result":"Task completed!","session_id":"test-123"}"""

        parser = ClaudeStreamParser()
        result = parser.parse_stream(output)

        assert result.success
        assert result.final_answer == "Task completed!"
        assert len(result.steps) >= 1

        # Check that tool result step was created with combined text
        tool_result_steps = [
            step for step in result.steps if step.step_type == "tool_result"
        ]
        assert len(tool_result_steps) == 1
        assert (
            "Multiple content items here and more text."
            in tool_result_steps[0].tool_output
        )

    @pytest.mark.asyncio
    async def test_claude_stream_parser_system_message(self):
        """Test parsing system initialization message with tools and MCP servers."""
        from smolval.claude_parser import ClaudeStreamParser

        # Claude output with system initialization
        output = """{"type":"system","subtype":"init","cwd":"/workspace","tools":["Read","Write","Bash"],"mcp_servers":[{"name":"sqlite","command":"docker"},{"name":"github","command":"npx"}],"model":"claude-sonnet-4","session_id":"test-123"}
{"type":"result","subtype":"success","result":"System initialized!","session_id":"test-123"}"""

        parser = ClaudeStreamParser()
        result = parser.parse_stream(output)

        assert result.success
        assert result.final_answer == "System initialized!"
        assert len(result.steps) >= 1

        # Check that system init step was created with detailed content
        system_steps = [
            step for step in result.steps if step.step_type == "system_init"
        ]
        assert len(system_steps) == 1
        system_step = system_steps[0]

        # Verify content includes model, tools, and MCP servers
        assert "claude-sonnet-4" in system_step.content
        assert "Read, Write, Bash" in system_step.content
        assert "sqlite (docker)" in system_step.content
        assert "github (npx)" in system_step.content
        assert "/workspace" in system_step.content

        # Verify metadata is properly extracted
        assert result.metadata.tools_available == ["Read", "Write", "Bash"]
        assert result.metadata.mcp_servers_used == ["sqlite", "github"]

    @pytest.mark.asyncio
    async def test_run_timeout(self, agent):
        """Test agent run with timeout."""
        # Set a very short timeout
        agent.timeout_seconds = 0.001

        with patch.object(agent, "_run_claude_code") as mock_run:
            # Make the method take longer than timeout
            async def slow_run(*args):
                from datetime import datetime

                from smolval.models import ExecutionMetadata

                await asyncio.sleep(1)
                metadata = ExecutionMetadata(
                    session_id="test",
                    model_used="test",
                    execution_start=datetime.now(),
                    execution_end=datetime.now(),
                )
                return AgentResult(
                    success=True,
                    final_answer="",
                    steps=[],
                    total_iterations=0,
                    execution_time_seconds=1.0,
                    metadata=metadata,
                )

            mock_run.side_effect = slow_run

            result = await agent.run("test prompt", show_progress=False)

            assert not result.success
            assert "timed out" in result.error_message
            assert result.execution_time_seconds >= 0.001

    @pytest.mark.asyncio
    async def test_find_claude_executable_found(self, agent):
        """Test finding Claude executable when available."""
        with (
            patch("subprocess.run") as mock_run,
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
        ):

            # Mock which command finding claude
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "/usr/local/bin/claude\n"
            mock_run.return_value = mock_result

            # Ensure os.path.exists and os.access don't find real files
            mock_exists.return_value = False
            mock_access.return_value = False

            executable = agent._find_claude_executable()

            assert executable == "/usr/local/bin/claude"

    @pytest.mark.asyncio
    async def test_find_claude_executable_not_found(self, agent):
        """Test finding Claude executable when not available."""
        with (
            patch("subprocess.run") as mock_run,
            patch("shutil.which") as mock_which,
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
        ):

            # Mock all commands failing
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result
            mock_which.return_value = None
            mock_exists.return_value = False
            mock_access.return_value = False

            executable = agent._find_claude_executable()

            assert executable is None

    @pytest.mark.asyncio
    async def test_run_claude_code_not_found(self, agent):
        """Test running when Claude CLI is not found."""
        with patch.object(agent, "_find_claude_executable") as mock_find:
            mock_find.return_value = None

            result = await agent._run_claude_code("test prompt", show_progress=False)

            assert not result.success
            assert "Claude Code CLI not found" in result.error_message

    @pytest.mark.asyncio
    async def test_run_claude_code_success(self, agent):
        """Test successful Claude Code execution."""
        with (
            patch.object(agent, "_find_claude_executable") as mock_find,
            patch("asyncio.create_subprocess_shell") as mock_subprocess_shell,
        ):

            mock_find.return_value = "claude"  # This will trigger shell execution

            # Mock successful subprocess with realistic stream-json output
            mock_process = AsyncMock()
            mock_process.returncode = 0
            stream_json_output = (
                '{"type":"result","subtype":"success","result":"Success!"}\n'
            )
            mock_process.communicate.return_value = (stream_json_output.encode(), b"")
            mock_subprocess_shell.return_value = mock_process

            result = await agent._run_claude_code("test prompt", show_progress=False)

            assert result.success
            assert result.final_answer == "Success!"
            assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_run_claude_code_failure(self, agent):
        """Test failed Claude Code execution."""
        with (
            patch.object(agent, "_find_claude_executable") as mock_find,
            patch("asyncio.create_subprocess_shell") as mock_subprocess_shell,
        ):

            mock_find.return_value = "claude"  # This will trigger shell execution

            # Mock failed subprocess
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"Error occurred")
            mock_subprocess_shell.return_value = mock_process

            result = await agent._run_claude_code("test prompt", show_progress=False)

            assert not result.success
            assert "Error occurred" in result.error_message
