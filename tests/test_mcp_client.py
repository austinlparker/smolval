"""Tests for MCP client manager."""

import asyncio
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from smolval.config import MCPServerConfig
from smolval.mcp_client import MCPClientManager, MCPTool


def setup_mocks():
    """Helper to set up common mocks."""
    mock_client = AsyncMock()
    mock_process = Mock()
    return mock_client, mock_process


@pytest.fixture
def filesystem_server_config() -> MCPServerConfig:
    """Return filesystem server configuration."""
    return MCPServerConfig(
        name="filesystem",
        command=["python", "-m", "mcp_server_filesystem", "/tmp"],
        env={}
    )


@pytest.fixture
def web_server_config() -> MCPServerConfig:
    """Return web search server configuration.""" 
    return MCPServerConfig(
        name="web_search",
        command=["node", "web-search-server.js"],
        env={"API_KEY": "test_key"}
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
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            server_name="filesystem"
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
            server_name="filesystem"
        )
        
        llm_tool = tool.to_llm_tool()
        assert llm_tool["type"] == "function"
        assert llm_tool["function"]["name"] == "list_files"
        assert "description" in llm_tool["function"]


class TestMCPClientManager:
    """Test MCP client manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        manager = MCPClientManager()
        assert len(manager.clients) == 0
        assert len(manager.tools) == 0

    @pytest.mark.asyncio
    async def test_connect_server_success(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test successfully connecting to an MCP server."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio, \
             patch('smolval.mcp_client.subprocess.Popen') as mock_popen:
            
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock subprocess
            mock_process = Mock()
            mock_popen.return_value = mock_process
            
            # Mock tool discovery
            mock_tool = Mock()
            mock_tool.name = "read_file"
            mock_tool.description = "Read file"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
            
            mock_client.list_tools.return_value.tools = [mock_tool]
            
            await manager.connect(filesystem_server_config)
            
            assert "filesystem" in manager.clients
            assert len(manager.tools) == 1
            assert manager.tools[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_connect_server_connection_failure(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test handling server connection failure."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio:
            mock_stdio.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await manager.connect(filesystem_server_config)
            
            assert "filesystem" not in manager.clients

    @pytest.mark.asyncio
    async def test_connect_multiple_servers(self, filesystem_server_config: MCPServerConfig, 
                                      web_server_config: MCPServerConfig) -> None:
        """Test connecting to multiple MCP servers."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio, \
             patch('smolval.mcp_client.subprocess.Popen') as mock_popen:
            
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock subprocess
            mock_process = Mock()
            mock_popen.return_value = mock_process
            
            # Mock different tools for each server call
            fs_tool = Mock()
            fs_tool.name = "read_file"
            fs_tool.description = "Read file"
            fs_tool.inputSchema = {}
            
            web_tool = Mock()
            web_tool.name = "web_search"
            web_tool.description = "Search web"
            web_tool.inputSchema = {}
            
            # Return different tools for each call
            call_count = 0
            def mock_list_tools():
                nonlocal call_count
                result = Mock()
                if call_count == 0:
                    result.tools = [fs_tool]
                else:
                    result.tools = [web_tool]
                call_count += 1
                return result
            
            mock_client.list_tools.side_effect = mock_list_tools
            
            await manager.connect(filesystem_server_config)
            await manager.add_server(web_server_config)
            
            assert len(manager.clients) == 2
            assert "filesystem" in manager.clients
            assert "web_search" in manager.clients
            assert len(manager.tools) == 2  # Both tools should be discovered

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test successful tool execution."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio:
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock tool discovery
            mock_tool = Mock()
            mock_tool.name = "read_file"
            mock_tool.description = "Read file"
            mock_tool.inputSchema = {}
            
            mock_client.list_tools.return_value.tools = [mock_tool]
            
            # Mock tool execution
            mock_result = Mock()
            mock_result.content = [Mock(type="text", text="File contents here")]
            mock_client.call_tool.return_value = mock_result
            
            await manager.connect(filesystem_server_config)
            
            result = await manager.execute_tool("read_file", {"path": "/test.txt"})
            
            assert result.tool_name == "read_file"
            assert result.server_name == "filesystem"
            assert "File contents" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        """Test executing non-existent tool."""
        manager = MCPClientManager()
        
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await manager.execute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test tool execution failure."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio:
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock tool discovery
            mock_tool = Mock()
            mock_tool.name = "read_file"
            mock_tool.description = "Read file"
            mock_tool.inputSchema = {}
            
            mock_client.list_tools.return_value.tools = [mock_tool]
            
            # Mock tool execution failure
            mock_client.call_tool.side_effect = Exception("Tool execution failed")
            
            await manager.connect(filesystem_server_config)
            
            result = await manager.execute_tool("read_file", {"path": "/test.txt"})
            
            assert result.tool_name == "read_file"
            assert result.error is not None
            assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_get_available_tools(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test getting list of available tools."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio:
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock tool discovery
            read_tool = Mock()
            read_tool.name = "read_file"
            read_tool.description = "Read file"
            read_tool.inputSchema = {}
            
            write_tool = Mock()
            write_tool.name = "write_file"
            write_tool.description = "Write file"
            write_tool.inputSchema = {}
            
            mock_client.list_tools.return_value.tools = [read_tool, write_tool]
            
            await manager.connect(filesystem_server_config)
            
            tools = manager.get_available_tools()
            
            assert len(tools) == 2
            tool_names = [tool.name for tool in tools]
            assert "read_file" in tool_names
            assert "write_file" in tool_names

    @pytest.mark.asyncio
    async def test_close_connections(self, filesystem_server_config: MCPServerConfig) -> None:
        """Test closing all MCP connections."""
        manager = MCPClientManager()
        
        with patch('smolval.mcp_client.stdio_client') as mock_stdio:
            mock_client = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_client.list_tools.return_value.tools = []
            
            await manager.connect(filesystem_server_config)
            assert len(manager.clients) == 1
            
            await manager.close()
            
            # Verify cleanup
            assert len(manager.clients) == 0
            assert len(manager.tools) == 0