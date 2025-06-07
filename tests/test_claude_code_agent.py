"""Tests for ClaudeCodeAgent."""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smolval.agent import ClaudeCodeAgent, AgentResult
from smolval.config import Config, MCPServerConfig, LLMConfig, EvaluationConfig
from smolval.mcp_client import MCPClientManager


@pytest.fixture
def claude_code_config():
    """Create a config for Claude Code agent."""
    return Config(
        mcp_servers=[
            MCPServerConfig(
                name="filesystem",
                command=["npx", "@modelcontextprotocol/server-filesystem", "."],
                env={}
            )
        ],
        llm=LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key"
        ),
        evaluation=EvaluationConfig(
            agent_type="claude_code",
            timeout_seconds=60,
            max_iterations=10
        )
    )


@pytest.fixture
def mcp_manager():
    """Create a mock MCP client manager."""
    return MagicMock(spec=MCPClientManager)


class TestClaudeCodeAgent:
    """Test cases for ClaudeCodeAgent."""

    def test_init(self, claude_code_config, mcp_manager):
        """Test agent initialization."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        assert agent.config == claude_code_config
        assert agent.mcp_manager == mcp_manager

    @pytest.mark.asyncio
    async def test_parse_claude_code_output_simple_text(self, claude_code_config, mcp_manager):
        """Test parsing simple text output."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        # Claude Code format with result object
        output = '''{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello, world!"}]}}
{"type": "result", "result": "Hello, world!"}'''
        final_answer, steps = agent._parse_claude_code_output(output)
        
        assert final_answer == "Hello, world!"
        assert len(steps) == 1
        assert steps[0].thought == "Hello, world!"
        assert steps[0].iteration == 1

    @pytest.mark.asyncio
    async def test_parse_claude_code_output_tool_use(self, claude_code_config, mcp_manager):
        """Test parsing tool use output."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        # Claude Code format with tool use and result
        output = '''{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "tool1", "name": "read_file", "input": {"path": "test.txt"}}]}}
{"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "tool1", "content": "File contents here"}]}}
{"type": "assistant", "message": {"content": [{"type": "text", "text": "Task completed!"}]}}
{"type": "result", "result": "Task completed! File contents here"}'''
        
        final_answer, steps = agent._parse_claude_code_output(output)
        
        assert "Task completed!" in final_answer
        assert len(steps) == 2
        
        # First step should be tool use
        assert steps[0].action == "read_file"
        assert steps[0].action_input == {"path": "test.txt"}
        assert steps[0].observation == "File contents here"
        
        # Second step should be final text
        assert steps[1].thought == "Task completed!"

    @pytest.mark.asyncio
    async def test_parse_claude_code_output_invalid_json(self, claude_code_config, mcp_manager):
        """Test parsing output with invalid JSON."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        output = 'This is not JSON\nNeither is this\n'
        final_answer, steps = agent._parse_claude_code_output(output)
        
        # Should use the last step's thought as final answer
        assert "Neither is this" in final_answer
        assert len(steps) == 2
        assert steps[0].thought == "This is not JSON"
        assert steps[1].thought == "Neither is this"

    @pytest.mark.asyncio
    async def test_get_existing_mcp_servers_success(self, claude_code_config, mcp_manager):
        """Test getting existing MCP servers."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful subprocess
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"server1\nserver2\n", b"")
            mock_subprocess.return_value = mock_process
            
            servers = await agent._get_existing_mcp_servers()
            
            assert "server1" in servers
            assert "server2" in servers

    @pytest.mark.asyncio
    async def test_get_existing_mcp_servers_failure(self, claude_code_config, mcp_manager):
        """Test getting existing MCP servers when command fails."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock failed subprocess
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process
            
            servers = await agent._get_existing_mcp_servers()
            
            assert servers == set()

    @pytest.mark.asyncio
    async def test_run_timeout(self, claude_code_config, mcp_manager):
        """Test agent run with timeout."""
        # Set a very short timeout
        claude_code_config.evaluation.timeout_seconds = 0.001
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        with patch.object(agent, '_run_claude_code') as mock_run:
            # Make the method take longer than timeout
            async def slow_run(*args):
                await asyncio.sleep(1)
                return AgentResult(success=True, final_answer="", steps=[], total_iterations=0)
            
            mock_run.side_effect = slow_run
            
            result = await agent.run("test prompt")
            
            assert not result.success
            assert "timed out" in result.error
            assert result.execution_time_seconds == claude_code_config.evaluation.timeout_seconds

    @pytest.mark.asyncio 
    async def test_setup_mcp_servers(self, claude_code_config, mcp_manager):
        """Test MCP server setup."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        with patch.object(agent, '_get_existing_mcp_servers') as mock_get_existing:
            mock_get_existing.return_value = set()  # No existing servers
            
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                # Mock successful server addition
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (b"", b"")
                mock_subprocess.return_value = mock_process
                
                await agent._setup_mcp_servers()
                
                # Should have called claude mcp add
                mock_subprocess.assert_called()
                call_args = mock_subprocess.call_args[0]
                # Check that the executable path contains claude (could be full path)
                assert any("claude" in str(arg) for arg in call_args)
                assert "mcp" in call_args
                assert "add" in call_args

    @pytest.mark.asyncio
    async def test_cleanup_mcp_servers(self, claude_code_config, mcp_manager):
        """Test MCP server cleanup."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock subprocess for cleanup
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            await agent._cleanup_mcp_servers()
            
            # Should have called claude mcp remove
            mock_subprocess.assert_called()
            call_args = mock_subprocess.call_args[0]
            # Check that the executable path contains claude (could be full path)
            assert any("claude" in str(arg) for arg in call_args)
            assert "mcp" in call_args
            assert "remove" in call_args

    def test_get_tool_permissions(self, claude_code_config, mcp_manager):
        """Test getting tool permissions."""
        agent = ClaudeCodeAgent(claude_code_config, mcp_manager)
        
        allowed_tools, disallowed_tools = agent._get_tool_permissions()
        
        # Check that we get lists
        assert isinstance(allowed_tools, list)
        assert isinstance(disallowed_tools, list)
        assert len(allowed_tools) > 0
        assert len(disallowed_tools) > 0
        
        # Check some expected allowed tools
        assert "LS(**)" in allowed_tools
        assert "Read(**)" in allowed_tools
        assert "Bash(git status)" in allowed_tools
        
        # Check some expected disallowed tools
        assert "Bash(rm *)" in disallowed_tools
        assert "Bash(sudo *)" in disallowed_tools
        
        # Ensure no overlap between allowed and disallowed
        overlap = set(allowed_tools) & set(disallowed_tools)
        assert len(overlap) == 0, f"Found overlapping tools: {overlap}"