# smolval: MCP Server Evaluation Agent

## Overview
smolval is a lightweight, self-contained Python application for evaluating MCP (Model Context Protocol) servers using LLM agents. It provides a simple agent loop that can read evaluation prompts, connect to MCP servers, and output structured results.

## Architecture

### Core Components

1. **Agent Loop Engine** (`agent.py`)
   - Implements ReAct pattern (Reason + Act)
   - Manages conversation state and tool execution
   - Handles error recovery and retries

2. **MCP Client Manager** (`mcp_client.py`)
   - Manages connections to multiple MCP servers
   - Handles protocol communication and tool discovery
   - Provides unified interface for tool execution

3. **Evaluation Framework** (`evaluator.py`)
   - Loads evaluation prompts from files
   - Executes test scenarios against MCP servers
   - Collects and structures results

4. **Configuration System** (`config.py`)
   - YAML-based configuration for MCP servers
   - LLM provider settings
   - Evaluation parameters

5. **Results Handler** (`results.py`)
   - Structured output formatting (JSON, CSV, Markdown)
   - Performance metrics collection
   - Error logging and analysis

### Agent Loop Pattern

```python
def evaluate_prompt(prompt: str, mcp_tools: List[Tool]) -> EvaluationResult:
    """Simple ReAct loop for MCP evaluation"""
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        # Reason: Get LLM response with available tools
        response = llm.chat(messages, tools=mcp_tools)
        
        # Act: Execute any tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = execute_mcp_tool(tool_call)
                messages.append(tool_result_message(result))
        else:
            # Done reasoning, return result
            return EvaluationResult(response.content, messages)
```

## Configuration

### MCP Server Configuration (`config.yaml`)
```yaml
mcp_servers:
  - name: "filesystem"
    command: ["python", "-m", "mcp_server_filesystem", "/tmp"]
    env: {}
  - name: "web_search"
    command: ["node", "web-search-server.js"]
    env:
      API_KEY: "${WEB_SEARCH_API_KEY}"

llm:
  provider: "anthropic"  # or "openai"
  model: "claude-3-sonnet-20240229"
  temperature: 0.1

evaluation:
  timeout_seconds: 60
  max_iterations: 10
  output_format: "json"
```

### Evaluation Prompts (`prompts/`)
- One file per evaluation scenario
- Supports templating for dynamic values
- Clear success/failure criteria

## Usage Patterns

### Basic Evaluation
```bash
# Run single evaluation
smolval eval prompts/file_operations.txt

# Run all evaluations in directory
smolval eval prompts/

# Custom configuration
smolval eval prompts/search.txt --config custom.yaml
```

### Batch Processing
```bash
# Run multiple MCP servers against same prompt
smolval batch --servers fs,web --prompt prompts/integration.txt

# Compare server performance
smolval compare --baseline fs --test web --prompts prompts/
```

## Implementation Phases

### Phase 1: Core Agent Loop
- Basic ReAct implementation
- Single MCP server connection
- File-based prompt loading
- JSON output

### Phase 2: Multi-Server Support
- MCP client manager
- Configuration system
- Parallel server evaluation
- Result comparison

### Phase 3: Advanced Features
- Performance metrics
- Error analysis
- Result visualization
- CI/CD integration

## Key Design Decisions

1. **Simplicity First**: Keep the core loop minimal and composable
2. **MCP-Native**: Use official MCP Python SDK for protocol compliance
3. **Provider Agnostic**: Support multiple LLM providers through unified interface
4. **Observable**: Comprehensive logging and metrics for debugging
5. **Extensible**: Plugin architecture for custom evaluation metrics

## Development Environment

### Python Environment Management
- **uv**: Used for fast Python package and environment management
- **pyproject.toml**: Modern Python project configuration
- **Docker**: Containerized deployment for consistent environments

### Development Approach
- **Test-Driven Development (TDD)**: Write tests first, then implementation
- **pytest**: Testing framework with fixtures for MCP server mocking
- **pytest-asyncio**: For testing async MCP client operations
- **pytest-docker**: For integration tests with real MCP servers

### Docker Configuration
```dockerfile
FROM python:3.11-slim
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
ENTRYPOINT ["uv", "run", "smolval"]
```

## Dependencies

- `mcp` - Official MCP Python SDK
- `anthropic` / `openai` - LLM providers
- `pydantic` - Configuration and data validation
- `click` - CLI interface
- `pyyaml` - Configuration parsing
- `rich` - Enhanced terminal output
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support
- `pytest-docker` - Docker integration testing

## Success Metrics

1. **Functional**: Can successfully execute evaluation prompts against MCP servers
2. **Reliable**: Handles connection failures and protocol errors gracefully
3. **Fast**: Completes typical evaluations in under 30 seconds
4. **Clear**: Produces actionable results for MCP server developers
5. **Easy**: Simple setup and configuration for new users