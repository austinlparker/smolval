"""LLM client using datasette llm library for unified provider interface."""

import json
from typing import Any


import llm
from pydantic import BaseModel

from .config import LLMConfig
from .mcp_client import MCPTool


class LLMMessage(BaseModel):
    """A message in the conversation history."""
    role: str  # "user", "assistant", "tool", "system"
    content: str
    tool_call_id: str | None = None
    tool_calls: list['ToolCall'] | None = None
    name: str | None = None


class ToolCall(BaseModel):
    """A tool call made by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Response from the LLM."""
    content: str
    tool_calls: list[ToolCall] = []
    stop_reason: str | None = None
    token_usage: dict[str, int] | None = None
    raw_response: dict[str, Any] | None = None




class LLMClient:
    """Client for interacting with LLM providers using datasette llm."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config
        self.conversation = None

        # Get the model from llm registry
        try:
            if config.provider == "ollama":
                # For Ollama, use the model name directly
                self.model = llm.get_model(config.model)
                # Set base URL if provided
                if config.base_url:
                    self.model.base_url = config.base_url
            else:
                self.model = llm.get_model(config.model)
        except Exception as e:
            raise ValueError(f"Model '{config.model}' not found. Make sure the appropriate llm plugin is installed. Error: {e}")

        # Set API key if provided (not needed for Ollama)
        if config.api_key and config.provider != "ollama":
            if config.provider == "anthropic":
                self.model.key = config.api_key
            elif config.provider == "openai":
                self.model.key = config.api_key

        # Initialize conversation
        self.conversation = self.model.conversation()

    async def chat(self, messages: list[LLMMessage], tools: list[MCPTool] | None = None) -> LLMResponse:
        """Send a chat request to the LLM using conversation API."""
        # Build the complete conversation context for each call
        # This ensures tools instructions are always included and conversation history is preserved
        
        # Extract system prompt 
        system_prompt = None
        conversation_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                conversation_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                conversation_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                conversation_parts.append(f"Tool result (ID: {msg.tool_call_id}): {msg.content}")
        
        # Add tool instructions to system prompt if tools are available
        if tools:
            tools_instruction = self._build_tools_instruction(tools)
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{tools_instruction}"
            else:
                system_prompt = tools_instruction
        
        # Build the complete conversation prompt
        conversation_text = "\n\n".join(conversation_parts)
        
        try:
            # Use the conversation API but with complete context each time
            # Note: Ollama models don't support max_tokens parameter
            if self.config.provider == "ollama":
                response = self.conversation.prompt(
                    conversation_text,
                    system=system_prompt,
                    temperature=self.config.temperature,
                )
            else:
                response = self.conversation.prompt(
                    conversation_text,
                    system=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        # Parse response
        content = response.text()
        tool_calls = []

        # Parse tool calls from response content if tools are available
        if tools:
            tool_calls = self._parse_tool_calls_from_text(content)
            if tool_calls:
                # Clear content if we found tool calls
                content = ""

        # Get token usage using the proper .usage() method
        token_usage = None
        try:
            usage = response.usage()
            if usage:
                token_usage = {
                    "input_tokens": getattr(usage, 'input', 0),
                    "output_tokens": getattr(usage, 'output', 0),
                    "total_tokens": getattr(usage, 'input', 0) + getattr(usage, 'output', 0)
                }
        except (AttributeError, Exception) as e:
            # If usage() method doesn't exist or fails, token_usage remains None
            pass

        # Get raw response JSON using the proper .json() method
        raw_response = None
        try:
            raw_response = response.json()
            if raw_response is None:
                response_dict = dict(response.response_json)
                raw_response = response_dict
        except (AttributeError, Exception) as e:
            # If json() method doesn't exist or fails, raw_response remains None
            pass
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=None,
            token_usage=token_usage,
            raw_response=raw_response
        )

    def _build_tools_instruction(self, tools: list[MCPTool]) -> str:
        """Build instruction text for available tools."""
        if self.config.provider == "ollama" and "gemma" in self.config.model.lower():
            return self._build_gemma_tools_instruction(tools)
        else:
            return self._build_json_tools_instruction(tools)
    
    def _build_gemma_tools_instruction(self, tools: list[MCPTool]) -> str:
        """Build Gemma-specific tool instruction using tool_code blocks."""
        instruction = "You have access to the following Python functions:\n\n"
        
        for tool in tools:
            # Convert tool to Python function signature
            params = []
            if tool.input_schema.get('properties'):
                for param, details in tool.input_schema['properties'].items():
                    param_type = details.get('type', 'str')
                    # Convert JSON schema types to Python types
                    python_type = {'string': 'str', 'integer': 'int', 'number': 'float', 'boolean': 'bool'}.get(param_type, 'str')
                    params.append(f"{param}: {python_type}")
            
            param_str = ", ".join(params)
            instruction += f"def {tool.name}({param_str}):\n"
            instruction += f'    """{tool.description}"""\n\n'
        
        instruction += "To use a function, wrap your function call in ```tool_code``` blocks like this:\n"
        instruction += "```tool_code\n"
        instruction += "result = function_name(param1=\"value1\", param2=\"value2\")\n"
        instruction += "```\n\n"
        instruction += "Think step-by-step about which function to use and how to call it correctly."
        
        return instruction
    
    def _build_json_tools_instruction(self, tools: list[MCPTool]) -> str:
        """Build traditional JSON-based tool instruction."""
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

    def _parse_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Parse tool calls from LLM response text."""
        if self.config.provider == "ollama" and "gemma" in self.config.model.lower():
            return self._parse_gemma_tool_calls(text)
        else:
            return self._parse_json_tool_calls(text)
    
    def _parse_gemma_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse Gemma tool_code blocks."""
        import re
        tool_calls = []
        
        # Look for ```tool_code``` blocks
        pattern = r'```tool_code\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            match = match.strip()
            
            # Parse function call from Python code
            # Handle both: result = read_file(path="/tmp/test.txt") AND read_file("/tmp/test.txt")
            func_pattern_with_assignment = r'(\w+)\s*=\s*(\w+)\((.*?)\)'
            func_pattern_direct = r'^(\w+)\((.*?)\)$'
            
            func_match = re.search(func_pattern_with_assignment, match)
            if func_match:
                result_var, func_name, args_str = func_match.groups()
            else:
                func_match = re.search(func_pattern_direct, match)
                if func_match:
                    func_name, args_str = func_match.groups()
                else:
                    continue
            
            # Parse arguments from string
            arguments = {}
            if args_str.strip():
                # Handle both: path="/tmp/test.txt" AND "/tmp/test.txt"
                # First try named arguments
                arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
                arg_matches = re.findall(arg_pattern, args_str)
                
                if arg_matches:
                    for arg_name, arg_value in arg_matches:
                        arguments[arg_name] = arg_value
                else:
                    # Handle positional arguments - assume first parameter is 'path' for file operations
                    positional_pattern = r'["\']([^"\']*)["\']'
                    positional_matches = re.findall(positional_pattern, args_str)
                    if positional_matches and func_name in ['read_file', 'write_file', 'list_directory']:
                        arguments['path'] = positional_matches[0]
            
            if func_name:  # Only add if we successfully parsed a function name
                tool_calls.append(ToolCall(
                    id=f"call_{len(tool_calls)}",
                    name=func_name,
                    arguments=arguments
                ))
        
        return tool_calls
    
    def _parse_json_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse traditional JSON tool calls."""
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
