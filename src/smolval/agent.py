"""ReAct agent implementation for MCP server evaluation."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from smolval.config import Config
from smolval.mcp_client import MCPClientManager, MCPToolResult
from smolval.llm_client import LLMClient, LLMMessage, LLMResponse


class AgentStep(BaseModel):
    """Represents a single step in the agent's reasoning process."""
    
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


class AgentResult(BaseModel):
    """Result from running the agent on a task."""
    
    success: bool
    final_answer: str
    steps: List[AgentStep]
    total_iterations: int
    error: Optional[str] = None
    execution_time_seconds: float = 0.0


class Agent:
    """ReAct agent for evaluating MCP servers."""
    
    def __init__(self, config: Config, llm_client: LLMClient, mcp_manager: MCPClientManager) -> None:
        """Initialize the agent."""
        self.config = config
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.conversation_history: List[LLMMessage] = []
    
    async def run(self, prompt: str) -> AgentResult:
        """Run the agent on a given prompt using ReAct pattern."""
        start_time = time.time()
        steps: List[AgentStep] = []
        
        try:
            # Initialize conversation with system prompt and user prompt
            self._reset_conversation()
            self._add_system_prompt()
            self._add_message("user", prompt)
            
            iteration = 0
            max_iterations = self.config.evaluation.max_iterations
            
            while iteration < max_iterations:
                iteration += 1
                
                try:
                    # Get LLM response
                    print(f"DEBUG: Getting LLM response for iteration {iteration}")
                    response = await self._get_llm_response()
                    print(f"DEBUG: Got response: {type(response)}")
                    
                    # Parse thought from response
                    thought = response.content or ""
                    
                    # Check if LLM wants to use tools
                    if response.tool_calls:
                        # Execute tool calls
                        for tool_call in response.tool_calls:
                            try:
                                step = await self._execute_tool_call(
                                    iteration, thought, tool_call
                                )
                                steps.append(step)
                                
                                # Add tool result to conversation
                                tool_id = getattr(tool_call, 'id', None) or tool_call.get('id', 'unknown') if isinstance(tool_call, dict) else 'unknown'
                                self._add_tool_result(tool_id, step.observation or "")
                            except Exception as tool_error:
                                # Handle both ToolCall objects and dictionaries
                                if hasattr(tool_call, 'name'):
                                    action = tool_call.name
                                    action_input = tool_call.arguments
                                    tool_id = getattr(tool_call, 'id', 'unknown')
                                else:
                                    action = tool_call.get('name', 'unknown')
                                    action_input = tool_call.get('arguments', {})
                                    tool_id = tool_call.get('id', 'unknown')
                                
                                error_step = AgentStep(
                                    iteration=iteration,
                                    thought=thought,
                                    action=action,
                                    action_input=action_input,
                                    observation=f"Error executing tool: {str(tool_error)}"
                                )
                                steps.append(error_step)
                                self._add_tool_result(tool_id, error_step.observation or "")
                    else:
                        # No tool calls - this is the final answer
                        step = AgentStep(
                            iteration=iteration,
                            thought=thought,
                            action=None,
                            action_input=None,
                            observation=None
                        )
                        steps.append(step)
                        
                        execution_time = time.time() - start_time
                        return AgentResult(
                            success=True,
                            final_answer=thought,
                            steps=steps,
                            total_iterations=iteration,
                            execution_time_seconds=execution_time
                        )
                        
                except Exception as e:
                    # Handle LLM errors - this should terminate the agent
                    execution_time = time.time() - start_time
                    return AgentResult(
                        success=False,
                        final_answer="",
                        steps=steps,
                        total_iterations=iteration - 1,  # Don't count failed iteration
                        error=str(e),
                        execution_time_seconds=execution_time
                    )
            
            # Max iterations reached
            execution_time = time.time() - start_time
            return AgentResult(
                success=False,
                final_answer="",
                steps=steps,
                total_iterations=iteration,
                error=f"Maximum iterations ({max_iterations}) exceeded",
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            # Top-level error (initialization, etc.)
            execution_time = time.time() - start_time
            return AgentResult(
                success=False,
                final_answer="",
                steps=steps,
                total_iterations=0,
                error=str(e),
                execution_time_seconds=execution_time
            )
    
    async def _get_llm_response(self) -> LLMResponse:
        """Get response from LLM with available tools."""
        available_tools = self.mcp_manager.get_available_tools()
        
        # Call LLM with conversation history and available tools
        return await self.llm_client.chat(
            messages=self.conversation_history,
            tools=available_tools if available_tools else None
        )
    
    async def _execute_tool_call(self, iteration: int, thought: str, tool_call) -> AgentStep:
        """Execute a single tool call and return the step."""
        # Handle both ToolCall objects and dictionaries
        if hasattr(tool_call, 'name'):
            tool_name = tool_call.name
            arguments = tool_call.arguments
            tool_id = getattr(tool_call, 'id', 'unknown')
        else:
            # Handle dictionary format
            tool_name = tool_call.get('name', 'unknown')
            arguments = tool_call.get('arguments', {})
            tool_id = tool_call.get('id', 'unknown')
        
        try:
            # Execute the tool
            result = await self.mcp_manager.execute_tool(tool_name, arguments)
            
            # Format observation
            if result.error:
                observation = f"Tool execution failed: {result.error}"
            else:
                observation = result.content
            
            return AgentStep(
                iteration=iteration,
                thought=thought,
                action=tool_name,
                action_input=arguments,
                observation=observation
            )
            
        except Exception as e:
            return AgentStep(
                iteration=iteration,
                thought=thought,
                action=tool_name,
                action_input=arguments,
                observation=f"Unexpected error executing tool: {str(e)}"
            )
    
    def _reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history.clear()
    
    def _add_system_prompt(self) -> None:
        """Add system prompt for ReAct pattern."""
        system_prompt = """You are a helpful assistant that can use tools to accomplish tasks. 

When you need to use tools, they will be provided to you as function calls. Use them when they can help accomplish the task.

Think step by step about what you need to do and use the available tools as needed.

When you have enough information to provide a final answer, provide your response directly."""
        
        self._add_message("system", system_prompt)
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(LLMMessage(
            role=role,
            content=content
        ))
    
    def _add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Add tool result to conversation history."""
        self.conversation_history.append(LLMMessage(
            role="tool",
            content=result,
            tool_call_id=tool_call_id
        ))