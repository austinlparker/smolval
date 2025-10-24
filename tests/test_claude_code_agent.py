"""Tests for ClaudeCodeAgent."""

import asyncio
from unittest.mock import patch

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
        assert agent.timeout_seconds == 900

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
