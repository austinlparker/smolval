"""Integration tests using real LLM APIs (requires API keys)."""

import os
import tempfile
from pathlib import Path

import pytest

from smolval.agent import Agent
from smolval.config import Config, EvaluationConfig, LLMConfig, MCPServerConfig
from smolval.llm_client import LLMClient
from smolval.mcp_client import MCPClientManager

# Mark these tests to require real API keys
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api_keys,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


@pytest.fixture
def real_anthropic_config() -> LLMConfig:
    """Real Anthropic config requiring API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")

    return LLMConfig(
        provider="anthropic",
        model="anthropic/claude-3-haiku-20240307",  # Use full model name
        api_key=api_key,
        temperature=0.1,
        max_tokens=500,  # Keep costs low
    )


@pytest.fixture
def real_openai_config() -> LLMConfig:
    """Real OpenAI config requiring API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    return LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",  # Use cheaper model for testing
        api_key=api_key,
        temperature=0.1,
        max_tokens=500,
    )


@pytest.fixture
def temp_test_files():
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "test.txt").write_text(
            "This is a test file for integration testing."
        )
        (temp_path / "data.json").write_text(
            '{"name": "integration_test", "status": "active"}'
        )
        (temp_path / "readme.md").write_text(
            "# Test Directory\nThis directory contains test files."
        )

        yield temp_path


class TestRealLLMIntegration:
    """Integration tests using real LLM APIs."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable",
    )
    async def test_anthropic_with_mock_filesystem(
        self, real_anthropic_config, temp_test_files
    ):
        """Test real Anthropic LLM with mock filesystem server."""
        config = Config(
            mcp_servers=[
                MCPServerConfig(
                    name="filesystem",
                    command=[
                        "npx",
                        "@modelcontextprotocol/server-filesystem",
                        str(temp_test_files),
                    ],
                    env={},
                )
            ],
            llm=real_anthropic_config,
            evaluation=EvaluationConfig(max_iterations=5, timeout_seconds=30),
        )

        # Initialize components
        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        try:
            # Connect to real filesystem MCP server
            await mcp_manager.connect(config.mcp_servers[0])

            agent = Agent(config, llm_client, mcp_manager)

            # Test with a filesystem task using the real MCP server
            prompt = "Please list the files in the directory and read the contents of test.txt. Tell me what you find."
            result = await agent.run(prompt)

            # Verify the agent completed the task
            assert result.success
            assert len(result.steps) > 0
            assert (
                "test.txt" in result.final_answer.lower()
                or "integration testing" in result.final_answer.lower()
            )

        finally:
            await mcp_manager.close()

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable",
    )
    async def test_openai_simple_task(self, real_openai_config):
        """Test real OpenAI LLM with simple task (no MCP servers)."""
        config = Config(
            mcp_servers=[],  # No MCP servers for this test
            llm=real_openai_config,
            evaluation=EvaluationConfig(max_iterations=2, timeout_seconds=15),
            allow_empty_servers=True,
        )

        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        try:
            agent = Agent(config, llm_client, mcp_manager)

            # Simple math task that doesn't require tools
            result = await agent.run("What is 15 * 7? Please just give me the answer.")

            assert result.success
            assert "105" in result.final_answer

        finally:
            await mcp_manager.close()

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable",
    )
    async def test_real_llm_error_handling(self, real_anthropic_config):
        """Test error handling with real LLM."""
        # Use invalid max_tokens to trigger an error
        bad_config = LLMConfig(
            provider=real_anthropic_config.provider,
            model=real_anthropic_config.model,
            api_key=real_anthropic_config.api_key,
            max_tokens=0,  # Invalid value
        )

        config = Config(
            mcp_servers=[],
            llm=bad_config,
            evaluation=EvaluationConfig(max_iterations=1, timeout_seconds=10),
            allow_empty_servers=True,
        )

        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        try:
            agent = Agent(config, llm_client, mcp_manager)

            # This should fail due to invalid max_tokens
            result = await agent.run("Simple test")

            # Should handle the error gracefully
            assert not result.success
            assert result.error is not None

        finally:
            await mcp_manager.close()


class TestLocalModelIntegration:
    """Integration tests for local models (e.g., Ollama)."""

    @pytest.mark.skipif(
        not os.path.exists("/usr/local/bin/ollama"),
        reason="Ollama not installed locally",
    )
    async def test_ollama_integration(self):
        """Test with local Ollama model."""
        # This would test with a local model like Ollama
        # First check if Ollama is running and has models available

        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                pytest.skip("Ollama not running or accessible")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No Ollama models available")

        except Exception:
            pytest.skip("Cannot connect to Ollama")

        # Use the first available model
        model_name = models[0]["name"]

        Config(
            mcp_servers=[
                MCPServerConfig(
                    name="filesystem",
                    command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    env={},
                )
            ],
            llm=LLMConfig(
                provider="ollama",
                model=model_name,
                api_key="not-needed",
                temperature=0.1,
                max_tokens=100,
            ),
            evaluation=EvaluationConfig(max_iterations=2, timeout_seconds=30),
        )

        # This test would require adding Ollama support to the LLM client
        pytest.skip("Ollama integration not implemented in LLM client yet")


# Test configuration for running real API tests
def pytest_configure(config):
    """Add markers for real API tests."""
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring real API keys"
    )


# Instructions for running these tests
"""
To run these integration tests with real LLMs:

1. Set environment variables:
   export ANTHROPIC_API_KEY="your-key-here"
   export OPENAI_API_KEY="your-key-here"

2. Run specific tests:
   # All real LLM tests (will cost money!)
   uv run pytest tests/test_integration_real_llm.py -v

   # Only Anthropic tests
   uv run pytest tests/test_integration_real_llm.py::TestRealLLMIntegration::test_anthropic_with_mock_filesystem -v

   # Skip expensive tests
   uv run pytest -m "integration and not requires_api_keys"

3. Run with cost controls:
   # Use cheaper models and lower token limits
   # Tests are configured to use cheaper models (Claude Haiku, GPT-3.5-turbo)
   # and limited max_tokens to minimize costs
"""
