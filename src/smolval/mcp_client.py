"""MCP client manager for connecting to and managing MCP servers."""

import logging
import os
import shutil
import subprocess
import sys
from contextlib import AsyncExitStack, redirect_stderr
from io import StringIO
from typing import Any

from mcp import ClientSession, stdio_client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

from smolval.config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """Represents an MCP tool with its metadata."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    def to_llm_tool(self) -> dict[str, Any]:
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
    error: str | None = None
    metadata: dict[str, Any] = {}


class MCPClientManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self) -> None:
        """Initialize the MCP client manager."""
        self.clients: dict[str, dict[str, Any]] = {}
        self.tools: list[MCPTool] = []
        self.exit_stack = AsyncExitStack()

    async def connect(self, config: MCPServerConfig, debug: bool = False) -> None:
        """Connect to an MCP server."""
        try:
            logger.debug("Connecting to MCP server: %s", config.name)
            logger.debug("Command: %s", config.command)
            
            # Create server parameters for the MCP stdio client
            # The first element is the command, the rest are arguments
            command = config.command[0]
            args = config.command[1:] if len(config.command) > 1 else []

            # Handle npx command like the official example
            if command == "npx":
                command = shutil.which("npx") or "npx"
                logger.debug("Using npx command: %s", command)

            logger.debug("Full command with args: %s %s", command, ' '.join(args))

            # Create environment with additional variables to suppress output
            env_vars = {**os.environ, **config.env} if config.env else os.environ.copy()
            
            # Add variables to suppress verbose output for known servers (unless debug mode)
            if not debug:
                env_vars.update({
                    'QUIET': '1',
                    'SILENT': '1', 
                    'NO_DEBUG': '1',
                    'MCP_QUIET': '1'
                })
                logger.debug("Added quiet environment variables")
            else:
                logger.debug("Debug mode: not adding quiet environment variables")
            
            # For some servers, we need to suppress both stdout and stderr
            # But MCP protocol uses stdio, so we'll try stderr first
            stderr_setting = None if debug else subprocess.DEVNULL
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env_vars,
                stderr=stderr_setting
            )

            logger.debug("Created server parameters for %s", config.name)

            # Suppress stderr during MCP connection to hide server startup messages (unless debug)
            if debug:
                logger.debug("Debug mode: not suppressing stderr")
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport

                logger.debug("Got stdio transport for %s", config.name)

                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                logger.debug("Created client session for %s", config.name)
                
                await session.initialize()
                logger.debug("Initialized session for %s", config.name)
            else:
                stderr_capture = StringIO()
                
                # Use proper async context manager pattern from official example
                with redirect_stderr(stderr_capture):
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
            logger.info("Successfully connected to MCP server: %s", config.name)

        except Exception as e:
            # Clean up if connection failed
            logger.error("Failed to connect to MCP server %s: %s", config.name, e)
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

            logger.debug("Discovered %d tools for server %s", len(tools_response.tools), server_name)

        except Exception as e:
            # Log warning but don't fail the connection
            logger.warning("Failed to discover tools for %s: %s", server_name, e)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
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
            logger.debug("Executing tool %s on server %s with arguments: %s", tool_name, tool.server_name, arguments)
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
            # Log as debug instead of error since tool failures are often expected in evaluations
            logger.debug("Tool execution failed for %s on server %s: %s", tool_name, tool.server_name, e)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=tool.server_name,
                content="",
                error=str(e)
            )

    def _find_tool(self, tool_name: str) -> MCPTool | None:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_available_tools(self) -> list[MCPTool]:
        """Get list of all available tools across all servers."""
        return self.tools.copy()

    def get_llm_tools(self) -> list[dict[str, Any]]:
        """Get tools formatted for LLM function calling."""
        return [tool.to_llm_tool() for tool in self.tools]

    async def close(self) -> None:
        """Close all MCP connections and clean up resources."""
        logger.debug("Starting MCP client manager cleanup...")
        
        # Use the exit stack to properly clean up all contexts
        try:
            logger.debug("Closing exit stack with %d contexts", len(self.clients))
            await self.exit_stack.aclose()
            logger.debug("Exit stack closed successfully")
        except ExceptionGroup as eg:
            if str(eg).startswith("unhandled errors in a TaskGroup"):
                logger.debug("Ignoring TaskGroupError during exit_stack cleanup: %s", eg)
            else:
                logger.warning("Error closing exit stack: %s", eg)
        except Exception as e:
            logger.warning("Error closing exit stack: %s", e)

        # Clear state
        logger.debug("Clearing client state")
        self.clients.clear()
        self.tools.clear()
        logger.debug("MCP client manager cleanup completed")

    async def disconnect_all(self) -> None:
        """Alias for close() method for CLI compatibility."""
        await self.close()
