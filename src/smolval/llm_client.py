"""LLM client using datasette llm library for unified provider interface."""

import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel
import llm

from .config import LLMConfig
from .mcp_client import MCPTool


class LLMMessage(BaseModel):
    """A message in the conversation history."""
    role: str  # "user", "assistant", "tool", "system"
    content: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolCall(BaseModel):
    """A tool call made by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    """Response from the LLM."""
    content: str
    tool_calls: List[ToolCall] = []
    stop_reason: Optional[str] = None


class LLMClient:
    """Client for interacting with LLM providers using datasette llm."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config
        
        # Get the model from llm registry
        try:
            self.model = llm.get_model(config.model)
        except Exception as e:
            raise ValueError(f"Model '{config.model}' not found. Make sure the appropriate llm plugin is installed. Error: {e}")
        
        # Set API key if provided
        if config.api_key:
            if config.provider == "anthropic":
                self.model.key = config.api_key
            elif config.provider == "openai":
                self.model.key = config.api_key
    
    async def chat(self, messages: List[LLMMessage], tools: Optional[List[MCPTool]] = None) -> LLMResponse:
        """Send a chat request to the LLM."""
        # Extract system prompt and build full prompt
        system_prompt = None
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_messages.append(msg.content)
            elif msg.role == "assistant":
                # Include assistant messages in context
                user_messages.append(f"Assistant previously said: {msg.content}")
            elif msg.role == "tool":
                # Add tool results
                user_messages.append(f"Tool result (ID: {msg.tool_call_id}): {msg.content}")
        
        # Combine user messages
        user_content = "\n\n".join(user_messages)
        
        # Add tool instructions to system prompt if tools are available
        if tools:
            tools_instruction = self._build_tools_instruction(tools)
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{tools_instruction}"
            else:
                system_prompt = tools_instruction
        
        try:
            # Use simple prompt API without tools parameter
            response = self.model.prompt(
                user_content,
                system=system_prompt
            )
            
            content = response.text()
            tool_calls = []
            
            # Parse tool calls from response content if tools are available
            if tools:
                tool_calls = self._parse_tool_calls_from_text(content)
                if tool_calls:
                    # Clear content if we found tool calls
                    content = ""
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                stop_reason=None
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def _build_tools_instruction(self, tools: List[MCPTool]) -> str:
        """Build instruction text for available tools."""
        instruction = "You have access to the following tools:\n\n"
        
        for tool in tools:
            instruction += f"**{tool.name}**: {tool.description}\n"
            if tool.input_schema.get('properties'):
                instruction += "Parameters:\n"
                for param, details in tool.input_schema['properties'].items():
                    param_type = details.get('type', 'string')
                    param_desc = details.get('description', '')
                    instruction += f"  - {param} ({param_type}): {param_desc}\n"
            instruction += "\n"
        
        instruction += "To use a tool, respond with JSON in this exact format:\n"
        instruction += '{"tool_name": "tool_name", "arguments": {"param": "value"}}\n\n'
        instruction += "Only respond with JSON when you want to use a tool. Otherwise, respond normally."
        
        return instruction
    
    def _parse_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        """Parse tool calls from LLM response text."""
        tool_calls = []
        
        # Look for JSON in the response
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    tool_data = json.loads(line)
                    if "tool_name" in tool_data and "arguments" in tool_data:
                        tool_calls.append(ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=tool_data["tool_name"],
                            arguments=tool_data["arguments"]
                        ))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls