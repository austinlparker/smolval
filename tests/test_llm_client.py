"""Tests for LLM client functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smolval.config import LLMConfig
from smolval.llm_client import LLMClient, LLMResponse, LLMMessage, ToolCall
from smolval.mcp_client import MCPTool


@pytest.fixture
def anthropic_config():
    """Anthropic LLM configuration."""
    return LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="test-api-key",
        max_tokens=1000,
        temperature=0.7
    )


@pytest.fixture
def openai_config():
    """OpenAI LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-api-key",
        max_tokens=1000,
        temperature=0.7
    )


@pytest.fixture
def sample_tools():
    """Sample MCP tools for testing."""
    return [
        MCPTool(
            name="read_file",
            description="Read file contents",
            server_name="filesystem",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        ),
        MCPTool(
            name="list_files",
            description="List files in directory",
            server_name="filesystem",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            }
        )
    ]


class TestLLMClient:
    """Test LLM client functionality."""

    def test_create_anthropic_client(self, anthropic_config):
        """Test creating Anthropic client."""
        client = LLMClient(anthropic_config)
        assert client.config == anthropic_config
        assert client.provider == "anthropic"

    def test_create_openai_client(self, openai_config):
        """Test creating OpenAI client."""
        client = LLMClient(openai_config)
        assert client.config == openai_config
        assert client.provider == "openai"

    @pytest.mark.asyncio
    @patch('anthropic.AsyncAnthropic')
    async def test_anthropic_chat_without_tools(self, mock_anthropic, anthropic_config):
        """Test Anthropic chat without tools."""
        # Setup mock response
        mock_client = AsyncMock()
        mock_anthropic.return_value = mock_client
        
        mock_content_block = MagicMock()
        mock_content_block.text = "Hello! How can I help you?"
        
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Test
        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Hello!")]
        
        response = await client.chat(messages)
        
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls == []
        assert response.stop_reason == "end_turn"
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["max_tokens"] == 1000
        assert call_args["temperature"] == 0.7
        assert call_args["messages"] == [{"role": "user", "content": "Hello!"}]

    @pytest.mark.asyncio
    @patch('anthropic.AsyncAnthropic')
    async def test_anthropic_chat_with_tools(self, mock_anthropic, anthropic_config, sample_tools):
        """Test Anthropic chat with tool calls."""
        # Setup mock response with tool call
        mock_client = AsyncMock()
        mock_anthropic.return_value = mock_client
        
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "call_123"
        mock_tool_use.name = "read_file"
        mock_tool_use.input = {"path": "/test/file.txt"}
        # Remove text attribute for tool_use blocks
        del mock_tool_use.text
        
        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        mock_response.stop_reason = "tool_use"
        
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Test
        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Read the file /test/file.txt")]
        
        response = await client.chat(messages, tools=sample_tools)
        
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "/test/file.txt"}
        assert response.stop_reason == "tool_use"
        
        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 2

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_openai_chat_without_tools(self, mock_openai, openai_config):
        """Test OpenAI chat without tools."""
        # Setup mock response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = "Hello! How can I help you?"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        client = LLMClient(openai_config)
        messages = [LLMMessage(role="user", content="Hello!")]
        
        response = await client.chat(messages)
        
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls == []
        assert response.stop_reason == "stop"

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_openai_chat_with_tools(self, mock_openai, openai_config, sample_tools):
        """Test OpenAI chat with tool calls."""
        # Setup mock response with tool call
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "read_file"
        mock_tool_call.function.arguments = '{"path": "/test/file.txt"}'
        
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test
        client = LLMClient(openai_config)
        messages = [LLMMessage(role="user", content="Read the file /test/file.txt")]
        
        response = await client.chat(messages, tools=sample_tools)
        
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "/test/file.txt"}
        assert response.stop_reason == "tool_calls"

    def test_convert_tools_to_anthropic_format(self, anthropic_config, sample_tools):
        """Test converting MCP tools to Anthropic format."""
        client = LLMClient(anthropic_config)
        tools = client._convert_tools_to_anthropic_format(sample_tools)
        
        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"
        assert tools[0]["description"] == "Read file contents"
        assert tools[0]["input_schema"]["type"] == "object"
        assert "path" in tools[0]["input_schema"]["properties"]

    def test_convert_tools_to_openai_format(self, openai_config, sample_tools):
        """Test converting MCP tools to OpenAI format."""
        client = LLMClient(openai_config)
        tools = client._convert_tools_to_openai_format(sample_tools)
        
        assert len(tools) == 2
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "read_file"
        assert tools[0]["function"]["description"] == "Read file contents"
        assert tools[0]["function"]["parameters"]["type"] == "object"

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises error."""
        # This test verifies that our provider validation works
        # The LLMConfig should raise a validation error for invalid provider
        with pytest.raises(ValueError, match="Provider must be 'anthropic' or 'openai'"):
            LLMConfig(
                provider="invalid",  # This should be caught by Pydantic validation
                model="test-model",
                api_key="test-key"
            )