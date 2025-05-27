"""Tests for the ReAct agent loop."""

from unittest.mock import AsyncMock, Mock

import pytest

from smolval.agent import Agent, AgentResult, AgentStep
from smolval.config import Config, EvaluationConfig, LLMConfig
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
        llm=LLMConfig(
            provider="anthropic", model="claude-sonnet-4-20250514", api_key="test-key"
        ),
        evaluation=EvaluationConfig(max_iterations=5, timeout_seconds=30),
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
            observation="File contents here",
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
            observation=None,
        )

        assert step.iteration == 3
        assert step.action is None
        assert step.observation is None


class TestAgentResult:
    """Test agent result representation."""

    def test_result_creation(self) -> None:
        """Test creating an agent result."""
        steps = [
            AgentStep(
                iteration=1,
                thought="First thought",
                action="action1",
                action_input={},
                observation="obs1",
            ),
            AgentStep(
                iteration=2,
                thought="Final thought",
                action=None,
                action_input=None,
                observation=None,
            ),
        ]

        result = AgentResult(
            success=True,
            final_answer="Task completed successfully",
            steps=steps,
            total_iterations=2,
            error=None,
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
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


class TestAgent:
    """Test the ReAct agent implementation."""

    def test_agent_initialization(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent initialization."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        assert agent.config == agent_config
        assert agent.llm_client == mock_llm_client
        assert agent.mcp_manager == mock_mcp_manager
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_simple_task_no_tools(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent with simple task requiring no tools."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Mock LLM response with no tool calls
        mock_response = Mock()
        mock_response.content = "The answer is 42."
        mock_response.tool_calls = []
        mock_response.token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.raw_response = {"id": "test-response"}
        mock_llm_client.chat = AsyncMock(return_value=mock_response)

        result = await agent.run("What is 2 + 2?")

        assert result.success is True
        assert result.final_answer == "The answer is 42."
        assert result.total_iterations == 1
        assert len(result.steps) == 1
        assert result.steps[0].action is None  # No tool used

    @pytest.mark.asyncio
    async def test_task_with_single_tool(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent with task requiring one tool call."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Set up available tools
        read_tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="filesystem",
        )
        mock_mcp_manager.get_available_tools.return_value = [read_tool]

        # Mock LLM responses
        from smolval.llm_client import ToolCall

        tool_call = ToolCall(
            id="call_1", name="read_file", arguments={"path": "/test.txt"}
        )

        responses = [
            # First response: decides to use tool
            Mock(
                content="I need to read the file",
                tool_calls=[tool_call],
                token_usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                raw_response={"id": "test-response-1"},
            ),
            # Second response: provides final answer
            Mock(
                content="Based on the file, the answer is 42.",
                tool_calls=[],
                token_usage={
                    "input_tokens": 15,
                    "output_tokens": 8,
                    "total_tokens": 23,
                },
                raw_response={"id": "test-response-2"},
            ),
        ]
        mock_llm_client.chat = AsyncMock(side_effect=responses)

        # Mock tool execution
        tool_result = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            content="File contains: important data",
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
    async def test_max_iterations_exceeded(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent hitting max iterations limit."""
        # Set very low max iterations
        agent_config.evaluation.max_iterations = 2
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Mock LLM always wants to use a tool
        from smolval.llm_client import ToolCall

        tool_call = ToolCall(id="call_1", name="some_tool", arguments={})

        mock_response = Mock()
        mock_response.content = "I need to use a tool"
        mock_response.tool_calls = [tool_call]
        mock_response.token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.raw_response = {"id": "test-response"}
        mock_llm_client.chat = AsyncMock(return_value=mock_response)

        # Mock tool execution
        tool_result = MCPToolResult(
            tool_name="some_tool", server_name="test", content="Tool result"
        )

        async def mock_execute_tool(*args, **kwargs):
            return tool_result

        mock_mcp_manager.execute_tool = mock_execute_tool

        result = await agent.run("Do something complex")

        assert result.success is False
        assert "maximum iterations" in result.error.lower()
        assert result.total_iterations == 2

    @pytest.mark.asyncio
    async def test_tool_execution_error(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test handling tool execution errors."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Mock LLM wants to use tool
        from smolval.llm_client import ToolCall

        tool_call = ToolCall(id="call_1", name="failing_tool", arguments={})

        mock_response = Mock()
        mock_response.content = "I'll use a tool"
        mock_response.tool_calls = [tool_call]
        mock_response.token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.raw_response = {"id": "test-response-1"}

        # Mock LLM handles error and provides final answer
        final_response = Mock()
        final_response.content = "I encountered an error but can still answer."
        final_response.tool_calls = []
        final_response.token_usage = {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }
        final_response.raw_response = {"id": "test-response-2"}
        mock_llm_client.chat = AsyncMock(side_effect=[mock_response, final_response])

        # Mock tool execution failure
        tool_result = MCPToolResult(
            tool_name="failing_tool",
            server_name="test",
            content="",
            error="Tool execution failed",
        )

        async def mock_execute_tool(*args, **kwargs):
            return tool_result

        mock_mcp_manager.execute_tool = mock_execute_tool

        result = await agent.run("Use the failing tool")

        assert result.success is True
        assert result.total_iterations == 2
        assert "Tool execution failed" in result.steps[0].observation

    @pytest.mark.asyncio
    async def test_llm_client_error(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test handling LLM client errors."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Mock LLM client failure
        mock_llm_client.chat = AsyncMock(side_effect=Exception("API Error"))

        result = await agent.run("Simple question")

        assert result.success is False
        assert "API Error" in result.error
        assert result.total_iterations == 0

    @pytest.mark.asyncio
    async def test_malformed_tool_call(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test handling malformed tool calls from LLM."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Mock LLM with tool call that will fail during execution
        from smolval.llm_client import ToolCall

        # Create a tool call with valid format but will cause execution error
        tool_call = ToolCall(
            id="call_1",
            name="nonexistent_tool",  # Tool that doesn't exist
            arguments={},
        )

        mock_response = Mock()
        mock_response.content = "I'll use a tool"
        mock_response.tool_calls = [tool_call]
        mock_response.token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.raw_response = {"id": "test-response-1"}

        # Mock LLM recovery
        final_response = Mock()
        final_response.content = "I'll provide an answer without tools."
        final_response.tool_calls = []
        final_response.token_usage = {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }
        final_response.raw_response = {"id": "test-response-2"}
        mock_llm_client.chat = AsyncMock(side_effect=[mock_response, final_response])

        # Mock tool execution to raise an error for unknown tools
        async def mock_execute_tool(tool_name, arguments):
            raise ValueError(f"Tool '{tool_name}' not found")

        mock_mcp_manager.execute_tool = mock_execute_tool

        result = await agent.run("Test malformed call")

        assert result.success is True
        assert result.total_iterations == 2
        assert (
            "not found" in result.steps[0].observation.lower()
            or "error executing tool" in result.steps[0].observation.lower()
        )

    def test_conversation_history_management(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test conversation history is properly maintained."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Add some messages
        agent._add_message("user", "Hello")
        agent._add_message("assistant", "Hi there")
        agent._add_message("user", "How are you?")

        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[0].role == "user"
        assert agent.conversation_history[0].content == "Hello"
        assert agent.conversation_history[1].role == "assistant"
        assert agent.conversation_history[2].content == "How are you?"

    @pytest.mark.asyncio
    async def test_error_recovery_token_limit(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent error recovery for token limit errors with smart memory management."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Simulate a conversation that's already long enough to need truncation
        for i in range(5):
            agent._add_message("user", f"Previous message {i}")
            agent._add_message("assistant", f"Previous response {i}")

        # First call raises token limit error, second call succeeds with recovery
        responses = [
            Exception(
                "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 205141 tokens > 200000 maximum'}}"
            ),
            Mock(
                content="With the summarized context, I can now provide the answer: 42.",
                tool_calls=[],
                token_usage={
                    "input_tokens": 10,
                    "output_tokens": 12,
                    "total_tokens": 22,
                },
                raw_response={"id": "test-recovery-response"},
            ),
        ]
        mock_llm_client.chat = AsyncMock(side_effect=responses)

        result = await agent.run("What is the answer?")

        assert result.success is True
        assert (
            result.final_answer
            == "With the summarized context, I can now provide the answer: 42."
        )
        assert result.total_iterations == 2
        assert len(result.steps) == 2

        # Check recovery step was added with smart memory management
        recovery_step = result.steps[0]
        assert "smart memory management" in recovery_step.thought
        assert "intelligent summarization" in recovery_step.observation
        assert "preserve key findings" in recovery_step.observation

        # Check conversation was intelligently truncated (preserves structure)
        # Should have: system+task messages + summary + recent context
        assert len(agent.conversation_history) <= 5

        # Verify smart truncation preserved important context
        # Look for summary message
        summary_found = any(
            "summarized" in msg.content.lower() for msg in agent.conversation_history
        )
        assert summary_found, "Should contain a summary message"

        # Check successful recovery step
        success_step = result.steps[1]
        assert success_step.action is None
        assert "summarized context" in success_step.thought

    @pytest.mark.asyncio
    async def test_non_recoverable_error(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent fails immediately on non-recoverable errors."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # API authentication error - not recoverable
        error = Exception(
            "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'Invalid API key'}}"
        )
        mock_llm_client.chat = AsyncMock(side_effect=error)

        result = await agent.run("What is the answer?")

        assert result.success is False
        assert "Invalid API key" in result.error
        assert result.total_iterations == 0  # Failed immediately
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_proactive_tool_size_management(
        self, agent_config: Config, mock_llm_client: Mock, mock_mcp_manager: Mock
    ) -> None:
        """Test agent proactively prevents oversized tool calls."""
        agent = Agent(agent_config, mock_llm_client, mock_mcp_manager)

        # Simulate a conversation with enough content to trigger proactive management
        for _ in range(10):
            agent._add_message("user", "x" * 1000)  # Add some conversation history
            agent._add_message("assistant", "y" * 1000)

        # Set a lower token budget for testing to trigger the proactive management
        agent._test_token_budget = 50000  # Much lower limit to ensure trigger

        # Mock LLM requesting a tool that would exceed limits
        from smolval.llm_client import ToolCall

        # Create a large database query that would exceed token limits
        large_query_tool = ToolCall(
            id="call_1",
            name="query",  # This tool is estimated to return 100k tokens
            arguments={"sql": "SELECT * FROM large_table"},
        )

        mock_response = Mock()
        mock_response.content = "I'll query the database to get all the data"
        mock_response.tool_calls = [large_query_tool]
        mock_response.token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.raw_response = {"id": "test-response-1"}

        # Second response after getting guidance
        refined_response = Mock()
        refined_response.content = (
            "Based on the guidance, I'll use a more focused query with LIMIT."
        )
        refined_response.tool_calls = []
        refined_response.token_usage = {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }
        refined_response.raw_response = {"id": "test-response-2"}

        mock_llm_client.chat = AsyncMock(side_effect=[mock_response, refined_response])

        # Mock execute_tool as async (shouldn't be called due to proactive management)
        # but set it up properly in case it does get called
        from smolval.mcp_client import MCPToolResult

        mock_result = MCPToolResult(
            tool_name="query",
            server_name="test",
            content="Unexpected execution",
            error="Mock should not have been called",
        )
        mock_mcp_manager.execute_tool = AsyncMock(return_value=mock_result)

        result = await agent.run("Get all database records")

        assert result.success is True
        assert result.total_iterations == 2
        assert len(result.steps) == 2

        # Check first step provided guidance instead of executing the tool
        guidance_step = result.steps[0]
        assert guidance_step.action == "query"
        assert "would likely return too much data" in guidance_step.observation
        assert "Add LIMIT clauses" in guidance_step.observation
        assert "SELECT *" in guidance_step.observation

        # Check LLM adjusted approach in second iteration
        final_step = result.steps[1]
        assert "more focused query" in final_step.thought

        # Verify that the tool was NOT called due to proactive management
        mock_mcp_manager.execute_tool.assert_not_called()
