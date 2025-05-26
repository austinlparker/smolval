"""Tests for MCP client manager."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from smolval.config import MCPServerConfig
from smolval.mcp_client import MCPClientManager, MCPTool, MCPToolResult


@pytest.fixture
def filesystem_server_config() -> MCPServerConfig:
    """Return filesystem server configuration."""
    return MCPServerConfig(
        name="filesystem", command=["python", "-m", "test_server"], env={}
    )


@pytest.fixture
def web_server_config() -> MCPServerConfig:
    """Return web search server configuration."""
    return MCPServerConfig(
        name="web_search",
        command=["node", "web-server.js"],
        env={"API_KEY": "test_key"},
    )


class TestMCPTool:
    """Test MCP tool representation."""

    def test_tool_creation(self) -> None:
        """Test creating an MCP tool."""
        tool = MCPTool(
            name="read_file",
            description="Read a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
            server_name="filesystem",
        )

        assert tool.name == "read_file"
        assert tool.server_name == "filesystem"
        assert "path" in tool.input_schema["properties"]

    def test_tool_call_format(self) -> None:
        """Test tool call formatting for LLM."""
        tool = MCPTool(
            name="list_files",
            description="List files in directory",
            input_schema={"type": "object", "properties": {}},
            server_name="filesystem",
        )

        llm_tool = tool.to_llm_tool()
        assert llm_tool["type"] == "function"
        assert llm_tool["function"]["name"] == "list_files"
        assert "description" in llm_tool["function"]

    def test_tool_with_empty_schema(self) -> None:
        """Test tool with no input schema."""
        tool = MCPTool(
            name="simple_tool",
            description="A simple tool",
            input_schema={},
            server_name="test",
        )

        assert tool.input_schema == {}
        llm_tool = tool.to_llm_tool()
        assert llm_tool["function"]["parameters"] == {}


class TestMCPToolResult:
    """Test MCP tool result representation."""

    def test_successful_result(self) -> None:
        """Test creating a successful tool result."""
        result = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            content="File contents here",
        )

        assert result.tool_name == "read_file"
        assert result.server_name == "filesystem"
        assert result.content == "File contents here"
        assert result.error is None
        assert result.metadata == {}

    def test_error_result(self) -> None:
        """Test creating an error tool result."""
        result = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            content="",
            error="File not found",
        )

        assert result.error == "File not found"
        assert result.content == ""

    def test_result_with_metadata(self) -> None:
        """Test tool result with metadata."""
        result = MCPToolResult(
            tool_name="test_tool",
            server_name="test",
            content="result",
            metadata={"execution_time": 1.5},
        )

        assert result.metadata["execution_time"] == 1.5


class TestMCPClientManager:
    """Test MCP client manager functionality."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        manager = MCPClientManager()
        assert len(manager.clients) == 0
        assert len(manager.tools) == 0
        assert manager.exit_stack is not None

    @pytest.mark.asyncio
    async def test_connect_server_success(
        self, filesystem_server_config: MCPServerConfig
    ) -> None:
        """Test successfully connecting to an MCP server."""
        manager = MCPClientManager()

        with (
            patch("smolval.mcp_client.stdio_client") as mock_stdio_client,
            patch("smolval.mcp_client.ClientSession") as mock_session_class,
        ):

            # Mock stdio_client to return transport
            mock_transport = (Mock(), Mock())  # (read, write)
            mock_stdio_client.return_value = mock_transport

            # Mock exit_stack.enter_async_context behavior
            async def mock_enter_context(context_manager):
                if context_manager == mock_transport:
                    return mock_transport
                else:
                    # This is the ClientSession
                    return mock_session

            manager.exit_stack.enter_async_context = AsyncMock(
                side_effect=mock_enter_context
            )

            # Mock ClientSession
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock tool discovery
            mock_tool = Mock()
            mock_tool.name = "read_file"
            mock_tool.description = "Read file"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }

            mock_tools_response = Mock()
            mock_tools_response.tools = [mock_tool]
            mock_session.list_tools.return_value = mock_tools_response

            await manager.connect(filesystem_server_config)

            assert "filesystem" in manager.clients
            assert len(manager.tools) == 1
            assert manager.tools[0].name == "read_file"
            assert manager.tools[0].server_name == "filesystem"

            # Verify session was initialized
            mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_server_with_npx_command(self) -> None:
        """Test connecting with npx command."""
        config = MCPServerConfig(
            name="npm_server",
            command=["npx", "@modelcontextprotocol/server-filesystem"],
            env={},
        )

        manager = MCPClientManager()

        with (
            patch("smolval.mcp_client.stdio_client") as mock_stdio_client,
            patch("smolval.mcp_client.ClientSession") as mock_session_class,
            patch("smolval.mcp_client.shutil.which") as mock_which,
        ):

            # Mock shutil.which to return a path for npx
            mock_which.return_value = "/usr/local/bin/npx"

            # Mock stdio_client and session
            mock_transport = (Mock(), Mock())
            mock_stdio_client.return_value = mock_transport

            async def mock_enter_context(context_manager):
                if context_manager == mock_transport:
                    return mock_transport
                else:
                    return mock_session

            manager.exit_stack.enter_async_context = AsyncMock(
                side_effect=mock_enter_context
            )

            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock empty tools response
            mock_tools_response = Mock()
            mock_tools_response.tools = []
            mock_session.list_tools.return_value = mock_tools_response

            await manager.connect(config)

            # Verify which was called for npx
            mock_which.assert_called_once_with("npx")

            assert "npm_server" in manager.clients

    @pytest.mark.asyncio
    async def test_connect_server_connection_failure(
        self, filesystem_server_config: MCPServerConfig
    ) -> None:
        """Test handling server connection failure."""
        manager = MCPClientManager()

        with patch("smolval.mcp_client.stdio_client") as mock_stdio_client:
            mock_stdio_client.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await manager.connect(filesystem_server_config)

            assert "filesystem" not in manager.clients

    @pytest.mark.asyncio
    async def test_connect_server_tool_discovery_failure(
        self, filesystem_server_config: MCPServerConfig
    ) -> None:
        """Test handling tool discovery failure."""
        manager = MCPClientManager()

        with (
            patch("smolval.mcp_client.stdio_client") as mock_stdio_client,
            patch("smolval.mcp_client.ClientSession") as mock_session_class,
        ):

            # Mock successful connection
            mock_transport = (Mock(), Mock())
            mock_stdio_client.return_value = mock_transport

            async def mock_enter_context(context_manager):
                if context_manager == mock_transport:
                    return mock_transport
                else:
                    return mock_session

            manager.exit_stack.enter_async_context = AsyncMock(
                side_effect=mock_enter_context
            )

            # Mock ClientSession with failing tool discovery
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.list_tools.side_effect = Exception("Tool discovery failed")

            # Should not raise exception, just log warning
            await manager.connect(filesystem_server_config)

            # Connection should still succeed
            assert "filesystem" in manager.clients
            # But no tools should be discovered
            assert len(manager.tools) == 0

    @pytest.mark.asyncio
    async def test_execute_tool_success(
        self, filesystem_server_config: MCPServerConfig
    ) -> None:
        """Test successful tool execution."""
        manager = MCPClientManager()

        # Manually add a tool and client to avoid complex connection mocking
        test_tool = MCPTool(
            name="read_file",
            description="Read file",
            input_schema={},
            server_name="filesystem",
        )
        manager.tools.append(test_tool)

        # Mock client session
        mock_session = AsyncMock()
        manager.clients["filesystem"] = {"session": mock_session}

        # Mock tool execution
        mock_content_item = Mock()
        mock_content_item.text = "File contents here"

        mock_result = Mock()
        mock_result.content = [mock_content_item]
        mock_session.call_tool.return_value = mock_result

        result = await manager.execute_tool("read_file", {"path": "/test.txt"})

        assert isinstance(result, MCPToolResult)
        assert result.tool_name == "read_file"
        assert result.server_name == "filesystem"
        assert "File contents here" in result.content
        assert result.error is None

        # Verify tool was called with correct arguments
        mock_session.call_tool.assert_called_once_with(
            "read_file", {"path": "/test.txt"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        """Test executing non-existent tool."""
        manager = MCPClientManager()

        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await manager.execute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_tool_no_client(self) -> None:
        """Test executing tool when client not available."""
        manager = MCPClientManager()

        # Add tool but no client
        test_tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema={},
            server_name="missing_server",
        )
        manager.tools.append(test_tool)

        with pytest.raises(
            ValueError, match="No client available for server 'missing_server'"
        ):
            await manager.execute_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self) -> None:
        """Test tool execution failure."""
        manager = MCPClientManager()

        # Manually add a tool and client
        test_tool = MCPTool(
            name="failing_tool",
            description="A failing tool",
            input_schema={},
            server_name="test_server",
        )
        manager.tools.append(test_tool)

        # Mock client that raises an exception
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = Exception("Tool execution failed")
        manager.clients["test_server"] = {"session": mock_session}

        result = await manager.execute_tool("failing_tool", {"param": "value"})

        assert result.tool_name == "failing_tool"
        assert result.server_name == "test_server"
        assert result.error == "Tool execution failed"
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_execute_tool_with_data_content(self) -> None:
        """Test tool execution with data content instead of text."""
        manager = MCPClientManager()

        # Add tool and client
        test_tool = MCPTool(
            name="data_tool",
            description="Returns data",
            input_schema={},
            server_name="data_server",
        )
        manager.tools.append(test_tool)

        # Mock client session
        mock_session = AsyncMock()
        manager.clients["data_server"] = {"session": mock_session}

        # Mock result with data content
        mock_content_item = Mock()
        del mock_content_item.text  # No text attribute
        mock_content_item.data = {"key": "value"}

        mock_result = Mock()
        mock_result.content = [mock_content_item]
        mock_session.call_tool.return_value = mock_result

        result = await manager.execute_tool("data_tool", {})

        assert "{'key': 'value'}" in result.content

    def test_get_available_tools(self) -> None:
        """Test getting available tools."""
        manager = MCPClientManager()

        # Add some tools
        tool1 = MCPTool(
            name="tool1", description="Tool 1", input_schema={}, server_name="server1"
        )
        tool2 = MCPTool(
            name="tool2", description="Tool 2", input_schema={}, server_name="server2"
        )

        manager.tools.extend([tool1, tool2])

        tools = manager.get_available_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

        # Verify it returns a copy (not the original list)
        tools.append(
            MCPTool(
                name="tool3",
                description="Tool 3",
                input_schema={},
                server_name="server3",
            )
        )
        assert len(manager.get_available_tools()) == 2  # Original unchanged

    def test_get_llm_tools(self) -> None:
        """Test getting tools formatted for LLM."""
        manager = MCPClientManager()

        # Add a tool
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
            },
            server_name="test_server",
        )
        manager.tools.append(tool)

        llm_tools = manager.get_llm_tools()

        assert len(llm_tools) == 1
        assert llm_tools[0]["type"] == "function"
        assert llm_tools[0]["function"]["name"] == "test_tool"
        assert llm_tools[0]["function"]["description"] == "A test tool"

    def test_find_tool(self) -> None:
        """Test finding a tool by name."""
        manager = MCPClientManager()

        # Add tools
        tool1 = MCPTool(
            name="tool1", description="Tool 1", input_schema={}, server_name="server1"
        )
        tool2 = MCPTool(
            name="tool2", description="Tool 2", input_schema={}, server_name="server2"
        )

        manager.tools.extend([tool1, tool2])

        # Test finding existing tool
        found_tool = manager._find_tool("tool1")
        assert found_tool is not None
        assert found_tool.name == "tool1"

        # Test finding non-existent tool
        not_found = manager._find_tool("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_close_connections(self) -> None:
        """Test closing all connections."""
        manager = MCPClientManager()

        # Mock the exit stack
        manager.exit_stack = AsyncMock()

        # Add some mock state
        manager.clients = {"test": {"session": Mock()}}
        manager.tools = [
            MCPTool(
                name="test", description="test", input_schema={}, server_name="test"
            )
        ]

        await manager.close()

        # Verify cleanup
        manager.exit_stack.aclose.assert_called_once()
        assert len(manager.clients) == 0
        assert len(manager.tools) == 0

    @pytest.mark.asyncio
    async def test_disconnect_all_alias(self) -> None:
        """Test disconnect_all is an alias for close."""
        manager = MCPClientManager()

        # Mock the exit stack
        manager.exit_stack = AsyncMock()

        await manager.disconnect_all()

        # Should call the same cleanup as close()
        manager.exit_stack.aclose.assert_called_once()
