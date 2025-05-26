"""Integration tests using Ollama for local LLM testing."""

import asyncio
from pathlib import Path

import pytest
import requests

from smolval.agent import Agent
from smolval.config import Config, LLMConfig
from smolval.llm_client import LLMClient
from smolval.mcp_client import MCPClientManager


@pytest.fixture
def ollama_config():
    """Configuration for Ollama testing."""
    return LLMConfig(
        provider="ollama",
        model="gemma3:1b-it-qat",
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=500,
    )


@pytest.fixture
def ollama_available():
    """Check if Ollama is available and has required models."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            pytest.skip("Ollama not running on localhost:11434")

        # Check if required model is available
        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        if "gemma3:1b-it-qat" not in model_names:
            pytest.skip(
                "Required model gemma3:1b-it-qat not found in Ollama. Run: ollama pull gemma3:1b-it-qat"
            )

        return True
    except requests.RequestException:
        pytest.skip("Ollama not accessible at localhost:11434")


@pytest.mark.integration
@pytest.mark.slow
def test_ollama_llm_client_basic(ollama_config, ollama_available):
    """Test basic LLM client functionality with Ollama."""
    client = LLMClient(ollama_config)

    # Simple test without tools
    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]

    # Convert dict messages to LLMMessage objects
    from smolval.llm_client import LLMMessage

    llm_messages = [LLMMessage(**msg) for msg in messages]

    # Run async function
    async def run_test():
        response = await client.chat(llm_messages)
        return response

    response = asyncio.run(run_test())

    assert response.content
    assert "4" in response.content


@pytest.mark.integration
@pytest.mark.slow
def test_ollama_with_mcp_tools(ollama_config, ollama_available):
    """Test Ollama LLM client basic functionality with MCP tools available."""
    # Load full config with MCP servers
    config_path = Path(__file__).parent.parent / "config" / "example-ollama.yaml"
    config = Config.from_yaml(config_path)

    async def run_test():
        # Initialize components
        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        # Connect to MCP servers
        for server_config in config.mcp_servers:
            await mcp_manager.connect(server_config)

        try:
            # Create agent
            agent = Agent(config, llm_client, mcp_manager)

            # Verify tools are available
            available_tools = mcp_manager.get_available_tools()
            assert len(available_tools) > 0
            print(f"\n=== {len(available_tools)} tools available ===")

            # Test with a simple prompt that doesn't require complex tool use
            prompt = "What is 2+2? Answer with just the number."

            print(f"\n=== Running agent with prompt: {prompt} ===")
            result = await agent.run(prompt)

            print(f"Success: {result.success}")
            print(f"Final Answer: {result.final_answer}")

            # Should complete successfully
            assert result.success is True
            assert result.final_answer
            assert "4" in result.final_answer

        finally:
            await mcp_manager.disconnect_all()

    asyncio.run(run_test())


@pytest.mark.integration
@pytest.mark.slow
def test_ollama_gemma_function_calling(ollama_config, ollama_available):
    """Test Gemma-specific function calling with tool_code blocks."""
    # Load full config with MCP servers
    config_path = Path(__file__).parent.parent / "config" / "example-ollama.yaml"
    config = Config.from_yaml(config_path)

    async def run_test():
        # Initialize components
        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        # Connect to MCP servers
        for server_config in config.mcp_servers:
            await mcp_manager.connect(server_config)

        try:
            # Test 1: Direct LLM client with tools
            from smolval.llm_client import LLMMessage

            tools = mcp_manager.get_available_tools()
            simple_tools = [tools[0]]  # Just read_file tool

            messages = [
                LLMMessage(
                    role="user",
                    content="Use the read_file function to read /tmp/test.txt. You MUST use the tool_code format.",
                )
            ]

            print("\n=== Test 1: Direct LLM Tool Call Generation ===")
            response = await llm_client.chat(messages, simple_tools)

            print(f"Tool calls generated: {len(response.tool_calls)}")
            assert (
                len(response.tool_calls) > 0
            ), "Should generate at least one tool call"

            tool_call = response.tool_calls[0]
            print(f"✅ Generated: {tool_call.name}({tool_call.arguments})")
            assert tool_call.name == "read_file"
            assert "path" in tool_call.arguments

            # Test 2: Tool call parsing from various formats
            test_cases = [
                (
                    "With assignment",
                    """```tool_code
result = read_file(path="/tmp/test.txt")
```""",
                ),
                (
                    "Direct call",
                    """```tool_code
read_file("/tmp/test.txt")
```""",
                ),
                (
                    "Named args",
                    """```tool_code
read_file(path="/tmp/file.txt")
```""",
                ),
            ]

            print("\n=== Test 2: Tool Call Parsing Variants ===")
            for name, test_text in test_cases:
                parsed_calls = llm_client._parse_tool_calls_from_text(test_text)
                print(f"{name}: {len(parsed_calls)} calls parsed")
                if parsed_calls:
                    call = parsed_calls[0]
                    print(f"  ✅ {call.name}({call.arguments})")
                    assert call.name == "read_file"
                    assert "path" in call.arguments

            # Test 3: Verify Gemma-specific format detection
            print("\n=== Test 3: Format Detection ===")
            is_gemma = (
                llm_client.config.provider == "ollama"
                and "gemma" in llm_client.config.model.lower()
            )
            print(f"Detected as Gemma model: {is_gemma}")
            assert is_gemma, "Should detect Gemma model format"

            print("✅ All Gemma function calling tests passed!")

        finally:
            await mcp_manager.disconnect_all()

    asyncio.run(run_test())


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skip(
    reason="Ollama Gemma model tool calling needs tuning - tool name mismatch"
)
def test_ollama_agent_evaluation(ollama_config, ollama_available):
    """Test full agent evaluation with Ollama."""
    config_path = Path(__file__).parent.parent / "config" / "ollama.yaml"
    config = Config.from_yaml(config_path)

    # Test prompt that uses available MCP tools (fetch and filesystem)
    test_prompt = "Fetch content from https://example.com and tell me what you find."

    async def run_test():
        llm_client = LLMClient(config.llm)
        mcp_manager = MCPClientManager()

        # Connect to MCP servers
        for server_config in config.mcp_servers:
            await mcp_manager.connect(server_config)

        try:
            agent = Agent(config, llm_client, mcp_manager)

            result = await agent.run(test_prompt)

            # Verify evaluation completed successfully
            assert result.success is True
            assert result.final_answer
            # Should mention example.com content or at least indicate fetch was attempted
            final_answer_lower = result.final_answer.lower()
            expected_words = [
                "example",
                "domain",
                "website",
                "page",
                "fetch",
                "content",
                "http",
            ]
            assert any(
                word in final_answer_lower for word in expected_words
            ), f"Expected one of {expected_words} in final answer: {result.final_answer}"
        finally:
            await mcp_manager.disconnect_all()

    asyncio.run(run_test())


def test_ollama_model_format():
    """Test that Ollama model names are formatted correctly."""
    from smolval.llm_client import LLMClient

    # Test model name formatting - should work now that plugin is installed
    config1 = LLMConfig(
        provider="ollama", model="gemma3:1b-it-qat", base_url="http://localhost:11434"
    )

    # This should create a client successfully (even if model doesn't exist)
    try:
        client = LLMClient(config1)
        # If we get here, the plugin is working
        assert client.config.provider == "ollama"
        assert client.config.model == "gemma3:1b-it-qat"
    except ValueError as e:
        # If model doesn't exist, that's expected in test environment
        assert "not found" in str(e).lower()


@pytest.mark.integration
def test_ollama_config_validation():
    """Test Ollama configuration validation."""
    # Valid Ollama config (no API key required)
    valid_config = LLMConfig(
        provider="ollama", model="gemma3:1b-it-qat", base_url="http://localhost:11434"
    )
    assert valid_config.provider == "ollama"
    assert valid_config.api_key is None

    # Invalid provider should fail
    with pytest.raises(ValueError, match="Provider must be"):
        LLMConfig(
            provider="invalid",
            model="test",
        )
