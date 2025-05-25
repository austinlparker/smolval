"""Tests for the ReAct agent loop."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from smolval.agent import Agent, AgentResult, AgentStep
from smolval.config import Config, LLMConfig, EvaluationConfig
from smolval.mcp_client import MCPTool, MCPToolResult


@pytest.fixture
def mock_llm_client() -> Mock:
    """Return a mock LLM client."""
    return Mock()


@pytest.fixture
def mock_mcp_manager() -> Mock:
    """Return a mock MCP client manager."""
    manager = Mock()
    manager.get_llm_tools.return_value = []
    return manager


@pytest.fixture
def agent_config() -> Config:
    """Return agent configuration."""
    from smolval.config import MCPServerConfig
    return Config(
        mcp_servers=[MCPServerConfig(name="test", command=["echo"], env={})],
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514", api_key="test-key"),
        evaluation=EvaluationConfig(max_iterations=5, timeout_seconds=30)
    )


class TestAgentStep:
    """Test agent step representation."""
    
    def test_step_creation(self) -> None:
        """Test creating an agent step."""
        step = AgentStep(
            iteration=1,
            thought="I need to read a file",
            action="read_file",
            action_input={"path": "/test.txt"},
            observation="File contents here"
        )
        
        assert step.iteration == 1
        assert step.thought == "I need to read a file"
        assert step.action == "read_file"
        assert step.observation == "File contents here"

    def test_step_without_action(self) -> None:
        """Test step with just thought (final reasoning)."""
        step = AgentStep(
            iteration=3,
            thought="Based on the file contents, I can conclude...",
            action=None,
            action_input=None,
            observation=None
        )
        
        assert step.iteration == 3
        assert step.action is None
        assert step.observation is None


class TestAgentResult:
    """Test agent result representation."""
    
    def test_result_creation(self) -> None:
        """Test creating an agent result."""
        steps = [
            AgentStep(iteration=1, thought="First thought", action="action1", action_input={}, observation="obs1"),
            AgentStep(iteration=2, thought="Final thought", action=None, action_input=None, observation=None)
        ]
        
        result = AgentResult(
            success=True,
            final_answer="Task completed successfully",
            steps=steps,
            total_iterations=2,
            error=None
        )
        
        assert result.success is True
        assert result.final_answer == "Task completed successfully"
        assert len(result.steps) == 2
        assert result.total_iterations == 2
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = AgentResult(
            success=False,
            final_answer="",
            steps=[],
            total_iterations=0,
            error="Connection timeout"
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"


class TestAgent:
    """Test the ReAct agent implementation."""
    
    def test_agent_initialization(self, agent_config: Config, mock_llm_client: Mock, 
                                 mock_mcp_manager: Mock) -> None:
        """Test agent initialization."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        assert agent.config == agent_config
        assert agent.llm_client == mock_llm_client
        assert agent.mcp_manager == mock_mcp_manager
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_simple_task_no_tools(self, agent_config: Config, mock_llm_client: Mock,
                                       mock_mcp_manager: Mock) -> None:
        """Test agent with simple task requiring no tools."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Mock LLM response with no tool calls
        mock_response = Mock()
        mock_response.content = "The answer is 42."
        mock_response.tool_calls = None
        mock_llm_client.chat_completion.return_value = mock_response
        
        result = await agent.run("What is 2 + 2?")
        
        assert result.success is True
        assert result.final_answer == "The answer is 42."
        assert result.total_iterations == 1
        assert len(result.steps) == 1
        assert result.steps[0].action is None  # No tool used

    @pytest.mark.asyncio
    async def test_task_with_single_tool(self, agent_config: Config, mock_llm_client: Mock,
                                        mock_mcp_manager: Mock) -> None:
        """Test agent with task requiring one tool call."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Set up available tools
        read_tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="filesystem"
        )
        mock_mcp_manager.get_available_tools.return_value = [read_tool]
        mock_mcp_manager.get_llm_tools.return_value = [read_tool.to_llm_tool()]
        
        # Mock LLM responses
        tool_function = Mock()
        tool_function.name = "read_file"
        tool_function.arguments = '{"path": "/test.txt"}'
        
        responses = [
            # First response: decides to use tool
            Mock(content="I need to read the file", tool_calls=[
                Mock(function=tool_function)
            ]),
            # Second response: provides final answer
            Mock(content="Based on the file, the answer is 42.", tool_calls=None)
        ]
        mock_llm_client.chat_completion.side_effect = responses
        
        # Mock tool execution
        tool_result = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            content="File contains: important data"
        )
        
        # Create async mock for execute_tool
        async def mock_execute_tool(*args, **kwargs):
            return tool_result
        
        mock_mcp_manager.execute_tool = mock_execute_tool
        
        result = await agent.run("What does the file /test.txt contain?")
        
        assert result.success is True
        assert result.final_answer == "Based on the file, the answer is 42."
        assert result.total_iterations == 2
        assert len(result.steps) == 2
        
        # Check first step (tool use)
        assert result.steps[0].action == "read_file"
        assert result.steps[0].observation == "File contains: important data"
        
        # Check second step (final answer)
        assert result.steps[1].action is None

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, agent_config: Config, mock_llm_client: Mock,
                                          mock_mcp_manager: Mock) -> None:
        """Test agent hitting max iterations limit."""
        # Set very low max iterations
        agent_config.evaluation.max_iterations = 2
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Mock LLM always wants to use a tool
        tool_function = Mock()
        tool_function.name = "some_tool"
        tool_function.arguments = '{}'
        
        mock_response = Mock()
        mock_response.content = "I need to use a tool"
        mock_response.tool_calls = [Mock(function=tool_function)]
        mock_llm_client.chat_completion.return_value = mock_response
        
        # Mock tool execution
        tool_result = MCPToolResult(
            tool_name="some_tool",
            server_name="test",
            content="Tool result"
        )
        
        async def mock_execute_tool(*args, **kwargs):
            return tool_result
        
        mock_mcp_manager.execute_tool = mock_execute_tool
        
        result = await agent.run("Do something complex")
        
        assert result.success is False
        assert "maximum iterations" in result.error.lower()
        assert result.total_iterations == 2

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, agent_config: Config, mock_llm_client: Mock,
                                       mock_mcp_manager: Mock) -> None:
        """Test handling tool execution errors."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Mock LLM wants to use tool
        tool_function = Mock()
        tool_function.name = "failing_tool"
        tool_function.arguments = '{}'
        
        mock_response = Mock()
        mock_response.content = "I'll use a tool"
        mock_response.tool_calls = [Mock(function=tool_function)]
        
        # Mock LLM handles error and provides final answer
        final_response = Mock()
        final_response.content = "I encountered an error but can still answer."
        final_response.tool_calls = None
        mock_llm_client.chat_completion.side_effect = [mock_response, final_response]
        
        # Mock tool execution failure
        tool_result = MCPToolResult(
            tool_name="failing_tool",
            server_name="test",
            content="",
            error="Tool execution failed"
        )
        
        async def mock_execute_tool(*args, **kwargs):
            return tool_result
        
        mock_mcp_manager.execute_tool = mock_execute_tool
        
        result = await agent.run("Use the failing tool")
        
        assert result.success is True
        assert result.total_iterations == 2
        assert "Tool execution failed" in result.steps[0].observation

    @pytest.mark.asyncio
    async def test_llm_client_error(self, agent_config: Config, mock_llm_client: Mock,
                                   mock_mcp_manager: Mock) -> None:
        """Test handling LLM client errors."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Mock LLM client failure
        mock_llm_client.chat_completion.side_effect = Exception("API Error")
        
        result = await agent.run("Simple question")
        
        assert result.success is False
        assert "API Error" in result.error
        assert result.total_iterations == 0

    @pytest.mark.asyncio 
    async def test_malformed_tool_call(self, agent_config: Config, mock_llm_client: Mock,
                                      mock_mcp_manager: Mock) -> None:
        """Test handling malformed tool calls from LLM."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Mock LLM with malformed tool call
        tool_function = Mock()
        tool_function.name = "test_tool"
        tool_function.arguments = 'invalid json'
        
        mock_response = Mock()
        mock_response.content = "I'll use a tool"
        mock_response.tool_calls = [Mock(function=tool_function)]
        
        # Mock LLM recovery
        final_response = Mock()
        final_response.content = "I'll provide an answer without tools."
        final_response.tool_calls = None
        mock_llm_client.chat_completion.side_effect = [mock_response, final_response]
        
        result = await agent.run("Test malformed call")
        
        assert result.success is True
        assert result.total_iterations == 2
        assert "error parsing" in result.steps[0].observation.lower() or "invalid" in result.steps[0].observation.lower()

    def test_conversation_history_management(self, agent_config: Config, mock_llm_client: Mock,
                                           mock_mcp_manager: Mock) -> None:
        """Test conversation history is properly maintained."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)
        
        # Add some messages
        agent._add_message("user", "Hello")
        agent._add_message("assistant", "Hi there")
        agent._add_message("user", "How are you?")
        
        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[0]["content"] == "Hello"
        assert agent.conversation_history[1]["role"] == "assistant"
        assert agent.conversation_history[2]["content"] == "How are you?"