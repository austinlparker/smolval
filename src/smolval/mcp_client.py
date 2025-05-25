"""MCP client manager for connecting to and managing MCP servers."""

import asyncio
import os
import shutil
import subprocess
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, stdio_client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

from smolval.config import MCPServerConfig


class MCPTool(BaseModel):
    """Represents an MCP tool with its metadata."""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    def to_llm_tool(self) -> Dict[str, Any]:
        """Convert to LLM tool format for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }


class MCPToolResult(BaseModel):
    """Result from executing an MCP tool."""
    
    tool_name: str
    server_name: str
    content: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class MCPClientManager:
    """Manages connections to multiple MCP servers."""
    
    def __init__(self) -> None:
        """Initialize the MCP client manager."""
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.tools: List[MCPTool] = []
        self.exit_stack = AsyncExitStack()
    
    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to an MCP server."""
        try:
            # Create server parameters for the MCP stdio client
            # The first element is the command, the rest are arguments
            command = config.command[0]
            args = config.command[1:] if len(config.command) > 1 else []
            
            # Handle npx command like the official example
            if command == "npx":
                command = shutil.which("npx") or "npx"
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env={**os.environ, **config.env} if config.env else os.environ
            )
            
            # Use proper async context manager pattern from official example
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            # Store the session for tool execution
            self.clients[config.name] = {
                'session': session
            }
            
            # Discover available tools
            await self._discover_tools(config.name, session)
                
        except Exception as e:
            # Clean up if connection failed
            if config.name in self.clients:
                del self.clients[config.name]
            raise e
    
    async def _discover_tools(self, server_name: str, session: ClientSession) -> None:
        """Discover tools available on an MCP server."""
        try:
            tools_response = await session.list_tools()
            
            # Follow the official example pattern for processing tools response
            for tool in tools_response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema or {},
                    server_name=server_name
                )
                self.tools.append(mcp_tool)
                
        except Exception as e:
            # Log warning but don't fail the connection
            print(f"Warning: Failed to discover tools for {server_name}: {e}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Execute a tool on the appropriate MCP server."""
        # Find the tool
        tool = self._find_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Get the client for this tool's server
        client_info = self.clients.get(tool.server_name)
        if not client_info:
            raise ValueError(f"No client available for server '{tool.server_name}'")
        
        session = client_info['session']
        
        try:
            # Execute the tool
            result = await session.call_tool(tool_name, arguments)
            
            # Extract content from result
            content_parts = []
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    elif hasattr(content_item, 'data'):
                        content_parts.append(str(content_item.data))
            
            content = "\n".join(content_parts) if content_parts else str(result)
            
            return MCPToolResult(
                tool_name=tool_name,
                server_name=tool.server_name,
                content=content
            )
            
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=tool.server_name,
                content="",
                error=str(e)
            )
    
    def _find_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_available_tools(self) -> List[MCPTool]:
        """Get list of all available tools across all servers."""
        return self.tools.copy()
    
    def get_llm_tools(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM function calling."""
        return [tool.to_llm_tool() for tool in self.tools]
    
    async def close(self) -> None:
        """Close all MCP connections and clean up resources."""
        # Use the exit stack to properly clean up all contexts
        await self.exit_stack.aclose()
        
        # Clear state
        self.clients.clear()
        self.tools.clear()
    
    async def disconnect_all(self) -> None:
        """Alias for close() method for CLI compatibility."""
        await self.close()