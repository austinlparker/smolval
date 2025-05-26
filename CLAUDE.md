# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

smolval is a lightweight Python application for evaluating Model Context Protocol (MCP) servers using LLM agents. It implements a ReAct (Reason + Act) pattern to systematically test MCP server implementations through structured evaluation prompts.

## Development Commands

### Package Management
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with development dependencies
uv sync --all-extras
```

### Running the Application

#### Local Development
```bash
# Single evaluation
uv run python -m smolval.cli eval prompts/simple_test.txt

# Batch evaluation with custom config
uv run python -m smolval.cli batch prompts/ -c config/custom.yaml -o results/ --format json

# Compare servers
uv run python -m smolval.cli compare --baseline filesystem --test fetch prompts/ --format markdown

# Use Ollama for local testing (requires ollama running with gemma3:1b-it-qat model)
# Features Gemma-specific function calling with tool_code blocks
uv run python -m smolval.cli eval prompts/simple_test.txt -c config/ollama.yaml --format markdown
```

#### Docker
```bash
# Build the Docker image
docker build -t smolval .

# Run with Docker socket for MCP servers that use Docker (like SQLite)
# Claude 4 Sonnet (default):
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  -e ANTHROPIC_API_KEY \
  smolval eval prompts/simple_test.txt --format markdown

# GPT-4o Mini:
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  -e OPENAI_API_KEY \
  smolval eval prompts/simple_test.txt -c config/openai-gpt4.yaml --format markdown

# Run without Docker socket (filesystem and fetch servers only)
docker run --rm \
  -e ANTHROPIC_API_KEY \
  smolval eval prompts/simple_test.txt -c config/filesystem-only.yaml --format markdown
```

### Testing
```bash
# Run full test suite 
uv run pytest

# Run with coverage
uv run pytest --cov=smolval

# Run specific test file
uv run pytest tests/test_agent.py

# Run Ollama integration tests (requires Ollama running locally)
uv run pytest tests/test_integration_ollama.py -m integration
```

### Code Quality
```bash
# Type checking
uv run mypy src/

# Formatting
uv run black src/ tests/
uv run isort src/ tests/

# Linting
uv run ruff check src/ tests/
```

## Core Architecture

### Component Relationships
1. **CLI** (`cli.py`) → **Config** (`config.py`) → **Agent** (`agent.py`) workflow
2. **Agent** orchestrates **LLMClient** + **MCPClientManager** in ReAct loop
3. **MCPClientManager** (`mcp_client.py`) discovers tools from multiple MCP servers, presents unified interface
4. **LLMClient** (`llm_client.py`) provides unified interface for Anthropic Claude, OpenAI, and Ollama models with Gemma-specific function calling
5. **ResultsFormatter** (`results.py`) handles multi-format output with Jinja2 templating

### Key Design Patterns
- **ReAct Agent Loop**: Step-by-step execution with thought, action, observation cycles
- **Async/Await**: Throughout for concurrent MCP server connections
- **Pydantic Models**: Type safety and validation for all data structures
- **Unified LLM Interface**: Uses `datasette llm` library for consistent provider interface
- **Plugin Architecture**: Extensible for new LLM providers and output formats

### MCP Client Architecture
The `MCPClientManager` uses proper async context managers with `AsyncExitStack` to manage connections to multiple MCP servers. It follows the official MCP Python SDK patterns:
- Uses `StdioServerParameters` for server configuration
- Implements proper context manager lifecycle with `exit_stack.enter_async_context()`
- Discovers tools from all connected servers and presents unified interface to the agent

### Configuration System
- **YAML-based**: Environment variable expansion with `${VAR}` and `${VAR:-default}` syntax (loads `.env` via python-dotenv)
- **Default Config**: `config/no-api-keys.yaml` with filesystem, fetch, sqlite servers (Claude 4 Sonnet)
- **Alternative Configs**: `config/openai-gpt4.yaml` (GPT-4o Mini), `config/filesystem-only.yaml` (Claude 4 Sonnet, filesystem only), `config/ollama.yaml` (Ollama with gemma3:1b-it-qat)
- **Required Environment**: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (not required for Ollama)
- **LLM Library**: Uses `datasette llm` with automatic plugin loading for providers

### MCP Server Support
- **NPM-based servers**: `@modelcontextprotocol/server-filesystem`
- **Python-based servers**: `mcp-server-fetch` (web content retrieval via uvx)
- **Docker-based servers**: `mcp/sqlite` (requires Docker socket access)
- **Mixed deployment**: Can run NPM, Python, and Docker-based servers simultaneously

## Important Implementation Details

### Agent Execution Flow
The agent runs a ReAct loop with:
1. **Thought**: LLM reasoning about next action
2. **Action**: Tool call with arguments
3. **Observation**: Tool execution result
4. Continues until task completion or max iterations (15)

### Tool Call Processing
Tool calls from LLM responses are processed as Pydantic `ToolCall` objects, then executed via the appropriate MCP server through `MCPClientManager.execute_tool()`.

### Testing Infrastructure
- **Real MCP Server Integration**: Tests use actual NPM and Python MCP servers
- **Mock LLM Responses**: Predefined responses to avoid API costs in testing
- **Async Testing**: Uses `pytest-asyncio` for async test functions
- **Multi-Provider Testing**: Tests with Anthropic, OpenAI, and Ollama LLM providers
- **Conditional Skipping**: Tests skip gracefully when dependencies aren't available

### Output Formats
Results can be formatted as JSON, CSV, Markdown, or HTML using Jinja2 templates in the `templates/` directory.