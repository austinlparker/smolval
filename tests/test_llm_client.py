"""Tests for LLM client functionality."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from smolval.config import LLMConfig
from smolval.llm_client import LLMClient, LLMMessage, LLMResponse
from smolval.mcp_client import MCPTool


def create_mock_response(text: str, input_tokens: int = 10, output_tokens: int = 5):
    """Helper function to create properly mocked LLM responses."""
    mock_response = Mock()
    mock_response.text.return_value = text
    mock_response.usage.return_value = Mock(input=input_tokens, output=output_tokens)
    mock_response.json.return_value = {
        "id": "test-response",
        "content": text,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    return mock_response


@pytest.fixture
def anthropic_config():
    """Anthropic LLM configuration."""
    return LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",  # Use a known model name
        api_key="test-api-key",
        max_tokens=1000,
        temperature=0.7,
    )


@pytest.fixture
def openai_config():
    """OpenAI LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-api-key",
        max_tokens=1000,
        temperature=0.7,
    )


@pytest.fixture
def gemini_config():
    """Gemini LLM configuration."""
    return LLMConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="test-api-key",
        max_tokens=1000,
        temperature=0.7,
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
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
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
                "required": ["path"],
            },
        ),
    ]


class TestLLMClient:
    """Test LLM client functionality."""

    @patch("llm.get_model")
    def test_create_anthropic_client(self, mock_get_model, anthropic_config):
        """Test creating Anthropic client."""
        # Mock the llm model
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        client = LLMClient(anthropic_config)
        assert client.config == anthropic_config
        assert hasattr(client, "model")

        # Verify the model was retrieved and key was set
        mock_get_model.assert_called_once_with("claude-3-sonnet-20240229")
        assert mock_model.key == "test-api-key"

    @patch("llm.get_model")
    def test_create_openai_client(self, mock_get_model, openai_config):
        """Test creating OpenAI client."""
        # Mock the llm model
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        client = LLMClient(openai_config)
        assert client.config == openai_config
        assert hasattr(client, "model")

        # Verify the model was retrieved and key was set
        mock_get_model.assert_called_once_with("gpt-4")
        assert mock_model.key == "test-api-key"

    @patch("llm.get_model")
    def test_model_not_found_raises_error(self, mock_get_model):
        """Test that unknown model raises appropriate error."""
        # Make get_model raise an exception
        mock_get_model.side_effect = Exception("Unknown model: nonexistent-model")

        config = LLMConfig(
            provider="anthropic", model="nonexistent-model", api_key="test-key"
        )

        with pytest.raises(ValueError, match="Model 'nonexistent-model' not found"):
            LLMClient(config)

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_chat_without_tools(self, mock_get_model, anthropic_config):
        """Test chat without tools."""
        # Setup mock model, conversation, and response
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        # Mock the response
        mock_response = create_mock_response("Hello! How can I help you?")
        mock_conversation.prompt.return_value = mock_response

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Hello!")]

        response = await client.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls == []
        assert response.stop_reason is None
        assert response.token_usage == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        assert response.raw_response == {
            "id": "test-response",
            "content": "Hello! How can I help you?",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        # Verify conversation prompt was called correctly
        mock_conversation.prompt.assert_called_once()
        call_args = mock_conversation.prompt.call_args
        assert "Hello!" in call_args[0][0]  # User content in prompt

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_chat_with_tools_no_calls(
        self, mock_get_model, anthropic_config, sample_tools
    ):
        """Test chat with tools available but no tool calls made."""
        # Setup mock model, conversation, and response
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        # Mock response with no tool calls
        mock_response = create_mock_response("I'll help you with that task.")
        mock_conversation.prompt.return_value = mock_response

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Hello!")]

        response = await client.chat(messages, tools=sample_tools)

        assert response.content == "I'll help you with that task."
        assert response.tool_calls == []

        # Verify tools instruction was added to prompt
        call_args = mock_conversation.prompt.call_args
        assert "read_file" in call_args[1]["system"]  # Tools should be in system prompt
        assert "You have access to the following tools" in call_args[1]["system"]

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_chat_with_tool_calls(
        self, mock_get_model, anthropic_config, sample_tools
    ):
        """Test chat with tool calls detected in response."""
        # Setup mock model, conversation, and response
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        # Mock response that contains tool call JSON
        tool_call_json = (
            '{"tool_name": "read_file", "arguments": {"path": "/test.txt"}}'
        )
        response_text = f"I'll read the file.\n{tool_call_json}"
        mock_response = create_mock_response(response_text)
        mock_conversation.prompt.return_value = mock_response

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Read /test.txt")]

        response = await client.chat(messages, tools=sample_tools)

        # Should detect and parse tool call
        assert response.content == ""  # Content cleared when tool calls found
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "/test.txt"}
        assert response.tool_calls[0].id.startswith("call_")

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_chat_with_multiple_tool_calls(
        self, mock_get_model, anthropic_config, sample_tools
    ):
        """Test chat with multiple tool calls in response."""
        # Setup mock model, conversation, and response
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        # Mock response with multiple tool calls
        response_text = """I'll help you with that.
{"tool_name": "list_files", "arguments": {"path": "/tmp"}}
{"tool_name": "read_file", "arguments": {"path": "/tmp/test.txt"}}"""

        mock_response = create_mock_response(response_text)
        mock_conversation.prompt.return_value = mock_response

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="List files then read test.txt")]

        response = await client.chat(messages, tools=sample_tools)

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "list_files"
        assert response.tool_calls[1].name == "read_file"
        assert response.tool_calls[0].id == "call_0"
        assert response.tool_calls[1].id == "call_1"

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_chat_with_invalid_json_ignored(
        self, mock_get_model, anthropic_config, sample_tools
    ):
        """Test that invalid JSON in response is ignored."""
        # Setup mock model, conversation, and response
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        # Mock response with invalid JSON
        response_text = """Here's what I found:
{"invalid": json syntax}
{"tool_name": "read_file", "arguments": {"path": "/test.txt"}}
{not json at all}"""

        mock_response = create_mock_response(response_text)
        mock_conversation.prompt.return_value = mock_response

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Help me")]

        response = await client.chat(messages, tools=sample_tools)

        # Should only parse the valid tool call
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    @patch("llm.get_model")
    async def test_llm_api_error_propagates(self, mock_get_model, anthropic_config):
        """Test that LLM API errors are properly propagated."""
        # Setup mock model, conversation that raises an error
        mock_model = Mock()
        mock_conversation = Mock()
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model
        mock_conversation.prompt.side_effect = Exception(
            "API Error: Rate limit exceeded"
        )

        client = LLMClient(anthropic_config)
        messages = [LLMMessage(role="user", content="Hello!")]

        with pytest.raises(RuntimeError, match="LLM API call failed"):
            await client.chat(messages)

    def test_build_tools_instruction(self, anthropic_config, sample_tools):
        """Test building tools instruction for system prompt."""
        with patch("llm.get_model") as mock_get_model:
            mock_model = Mock()
            mock_conversation = Mock()
            mock_model.conversation.return_value = mock_conversation
            mock_get_model.return_value = mock_model

            client = LLMClient(anthropic_config)
            instruction = client._build_tools_instruction(sample_tools)

            assert "You have access to the following tools" in instruction
            assert "read_file" in instruction
            assert "list_files" in instruction
            assert "Read file contents" in instruction
            assert "path (string)" in instruction
            assert (
                '{"tool_name": "tool_name", "arguments": {"param": "value"}}'
                in instruction
            )

    def test_parse_tool_calls_from_text(self, anthropic_config):
        """Test parsing tool calls from response text."""
        with patch("llm.get_model") as mock_get_model:
            mock_model = Mock()
            mock_conversation = Mock()
            mock_model.conversation.return_value = mock_conversation
            mock_get_model.return_value = mock_model

            client = LLMClient(anthropic_config)

            # Test valid tool call
            text = '{"tool_name": "read_file", "arguments": {"path": "/test.txt"}}'
            tool_calls = client._parse_tool_calls_from_text(text)

            assert len(tool_calls) == 1
            assert tool_calls[0].name == "read_file"
            assert tool_calls[0].arguments == {"path": "/test.txt"}

            # Test no tool calls
            text = "Just a regular response with no tool calls."
            tool_calls = client._parse_tool_calls_from_text(text)
            assert len(tool_calls) == 0

            # Test mixed content
            text = """Here's what I'll do:
{"tool_name": "list_files", "arguments": {"path": "/"}}
And then I'll analyze the results."""
            tool_calls = client._parse_tool_calls_from_text(text)
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "list_files"

    def test_message_history_formatting(self, anthropic_config):
        """Test how different message types are formatted with conversation API."""
        with patch("llm.get_model") as mock_get_model:
            mock_model = Mock()
            mock_conversation = Mock()
            mock_model.conversation.return_value = mock_conversation
            mock_response = create_mock_response("Response")
            mock_conversation.prompt.return_value = mock_response
            mock_get_model.return_value = mock_model

            client = LLMClient(anthropic_config)

            # Test subsequent turn with tool results
            messages = [
                LLMMessage(role="system", content="You are a helpful assistant"),
                LLMMessage(role="user", content="Hello"),
                LLMMessage(role="assistant", content="Hi there!"),
                LLMMessage(role="tool", content="File contents", tool_call_id="call_1"),
            ]

            # This should not raise an error and should format all message types
            asyncio.run(client.chat(messages))

            # Verify conversation prompt was called
            mock_conversation.prompt.assert_called_once()
            call_args = mock_conversation.prompt.call_args

            # For subsequent turns, should send tool results
            turn_content = call_args[0][0]
            assert "Tool result (ID: call_1): File contents" in turn_content

    def test_gemini_client_initialization(self, gemini_config):
        """Test Gemini client initialization."""
        with patch("llm.get_model") as mock_get_model:
            mock_model = Mock()
            mock_conversation = Mock()
            mock_model.conversation.return_value = mock_conversation
            mock_get_model.return_value = mock_model

            client = LLMClient(gemini_config)

            assert client.config.provider == "gemini"
            assert client.config.model == "gemini-2.0-flash"
            # Verify API key is set
            assert mock_model.key == "test-api-key"

    @patch("llm.get_model")
    def test_gemini_chat_request(self, mock_get_model, gemini_config):
        """Test Gemini chat request."""
        # Setup mocks
        mock_model = Mock()
        mock_conversation = Mock()
        mock_response = create_mock_response("Hello from Gemini!")

        mock_conversation.prompt.return_value = mock_response
        mock_model.conversation.return_value = mock_conversation
        mock_get_model.return_value = mock_model

        client = LLMClient(gemini_config)

        # Test message
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant"),
            LLMMessage(role="user", content="Hello!"),
        ]

        response = asyncio.run(client.chat(messages))

        # Verify response
        assert response.content == "Hello from Gemini!"
        assert response.token_usage is not None
        assert response.token_usage["total_tokens"] == 15

        # Verify conversation was called with correct parameters
        mock_conversation.prompt.assert_called_once()
        call_args = mock_conversation.prompt.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 1000

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises error."""
        with pytest.raises(
            ValueError,
            match="Provider must be 'anthropic', 'openai', 'ollama', or 'gemini'",
        ):
            LLMConfig(provider="invalid", model="test-model", api_key="test-key")
