"""Simple integration tests that can actually run."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from smolval.agent import Agent
from smolval.config import Config, EvaluationConfig, LLMConfig, MCPServerConfig
from smolval.llm_client import LLMClient, LLMResponse, ToolCall
from smolval.mcp_client import MCPClientManager, MCPTool, MCPToolResult


@pytest.mark.integration
class TestSimpleIntegration:
    """Simple integration tests using mocks that verify component integration."""

    @pytest.fixture
    def integration_config(self) -> Config:
        """Create a test configuration."""
        return Config(
            mcp_servers=[
                MCPServerConfig(name="test_server", command=["echo", "test"], env={})
            ],
            llm=LLMConfig(
                provider="anthropic",
                model="test-model",
                api_key="test-key",
                temperature=0.1,
            ),
            evaluation=EvaluationConfig(max_iterations=3, timeout_seconds=10),
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=LLMClient)
        return client

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager with tools."""
        manager = Mock(spec=MCPClientManager)

        # Mock available tools
        test_tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            server_name="test_server",
        )
        manager.get_available_tools.return_value = [test_tool]

        # Mock tool execution
        async def mock_execute_tool(tool_name, arguments):
            return MCPToolResult(
                tool_name=tool_name,
                server_name="test_server",
                content=f"Executed {tool_name} with {arguments}",
            )

        manager.execute_tool = mock_execute_tool
        return manager

    @pytest.mark.asyncio
    async def test_agent_integration_with_mocks(
        self, integration_config, mock_llm_client, mock_mcp_manager
    ):
        """Test agent integration with mocked components."""

        # Setup mock LLM responses
        responses = [
            LLMResponse(
                content="I'll use the test tool",
                tool_calls=[ToolCall(id="call_1", name="test_tool", arguments={})],
            ),
            LLMResponse(content="Task completed successfully", tool_calls=[]),
        ]
        mock_llm_client.chat = AsyncMock(side_effect=responses)

        # Create agent
        agent = Agent(integration_config, mock_llm_client, mock_mcp_manager)

        # Run test
        result = await agent.run("Test task")

        # Verify integration
        assert result.success
        assert result.total_iterations == 2
        assert len(result.steps) == 2
        assert result.steps[0].action == "test_tool"
        assert result.steps[1].action is None  # Final response

        # Verify LLM was called
        assert mock_llm_client.chat.call_count == 2

        # Verify MCP manager was used
        mock_mcp_manager.get_available_tools.assert_called()

    @pytest.mark.asyncio
    async def test_full_component_lifecycle(self, integration_config):
        """Test creating and connecting real components (without external dependencies)."""

        # Test config loading and validation
        assert len(integration_config.mcp_servers) == 1
        assert integration_config.llm.provider == "anthropic"
        assert integration_config.evaluation.max_iterations == 3

        # Test LLM client creation (will fail gracefully with mock model)
        try:
            llm_client = LLMClient(integration_config.llm)
            # If we get here, the client was created successfully
            # but calls will fail due to mock model
            assert llm_client.config == integration_config.llm
        except ValueError as e:
            # Expected for mock model
            assert "not found" in str(e)

        # Test MCP manager creation
        mcp_manager = MCPClientManager()
        assert len(mcp_manager.clients) == 0
        assert len(mcp_manager.tools) == 0

        # Test cleanup
        await mcp_manager.close()

    def test_config_validation_integration(self):
        """Test configuration validation across components."""

        # Test invalid config scenarios
        with pytest.raises(ValueError):
            Config(
                mcp_servers=[],  # Empty servers should fail
                llm=LLMConfig(provider="anthropic", model="test", api_key="test"),
                evaluation=EvaluationConfig(),
            )

        with pytest.raises(ValueError):
            Config(
                mcp_servers=[MCPServerConfig(name="test", command=["echo"], env={})],
                llm=LLMConfig(
                    provider="invalid", model="test", api_key="test"
                ),  # Invalid provider
                evaluation=EvaluationConfig(),
            )

    def test_results_formatting_integration(self):
        """Test results formatting with realistic data."""
        from smolval.agent import AgentResult, AgentStep
        from smolval.results import ResultsFormatter

        # Create realistic test data
        steps = [
            AgentStep(
                iteration=1,
                thought="I need to test something",
                action="test_tool",
                action_input={"param": "value"},
                observation="Tool executed successfully",
            ),
            AgentStep(
                iteration=2,
                thought="Task completed",
                action=None,
                action_input=None,
                observation=None,
            ),
        ]

        result = AgentResult(
            success=True,
            final_answer="Test completed successfully",
            steps=steps,
            total_iterations=2,
            execution_time_seconds=1.5,
        )

        result_data = {
            "prompt": "Test prompt",
            "result": {
                "success": result.success,
                "final_answer": result.final_answer,
                "steps": [step.model_dump() for step in result.steps],
                "total_iterations": result.total_iterations,
                "execution_time_seconds": result.execution_time_seconds,
                "error": result.error,
            },
            "metadata": {"timestamp": 1234567890, "config_file": "test.yaml"},
        }

        # Test different formatters
        for format_type in ["json", "csv", "markdown", "html"]:
            formatter = ResultsFormatter(format_type)
            output = formatter.format_single_result(result_data)
            assert output is not None
            assert len(output) > 0

            if format_type == "json":
                import json

                # Verify valid JSON
                parsed = json.loads(output)
                assert parsed["result"]["success"] is True


@pytest.mark.integration
@pytest.mark.slow
class TestFileSystemIntegration:
    """Integration tests that work with real file system."""

    def test_temp_file_operations(self):
        """Test file operations that integration tests might use."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            test_file = temp_path / "test.txt"
            test_file.write_text("Hello, integration test!")

            data_file = temp_path / "data.json"
            data_file.write_text('{"test": true}')

            # Verify files exist
            assert test_file.exists()
            assert data_file.exists()

            # Test reading
            content = test_file.read_text()
            assert "integration test" in content

            # Test listing
            files = list(temp_path.iterdir())
            assert len(files) == 2

            file_names = [f.name for f in files]
            assert "test.txt" in file_names
            assert "data.json" in file_names

    def test_prompt_file_creation(self):
        """Test creating and reading prompt files for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test prompt
            prompt_file = temp_path / "test_prompt.txt"
            prompt_content = """
            Please perform the following tasks:
            1. List files in the directory
            2. Read any text files you find
            3. Summarize the contents

            This is a test prompt for integration testing.
            """
            prompt_file.write_text(prompt_content.strip())

            # Verify prompt can be read
            loaded_content = prompt_file.read_text()
            assert "integration testing" in loaded_content
            assert "List files" in loaded_content


# Test utility functions for integration tests
def create_test_config(temp_dir: Path) -> Path:
    """Create a test configuration file."""
    config_content = f"""
mcp_servers:
  - name: "test_filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "{temp_dir}"]
    env: {{}}

llm:
  provider: "anthropic"
  model: "test-model"
  api_key: "test-key"
  temperature: 0.1

evaluation:
  timeout_seconds: 30
  max_iterations: 5
"""
    config_file = temp_dir / "test_config.yaml"
    config_file.write_text(config_content.strip())
    return config_file


def create_test_prompt(temp_dir: Path, task: str) -> Path:
    """Create a test prompt file."""
    prompt_content = f"""
Task: {task}

Instructions:
1. Use available MCP tools to complete this task
2. Provide clear explanations of what you're doing
3. Summarize your findings

Expected outcome: Successful completion with detailed explanation.
"""
    prompt_file = temp_dir / "test_prompt.txt"
    prompt_file.write_text(prompt_content.strip())
    return prompt_file
