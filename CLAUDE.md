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

# Compare different LLM providers (requires both plugins and API keys)
uv run python -m smolval.cli compare-providers --baseline-config config/example-anthropic.yaml --test-config config/example-gemini.yaml prompts/ --format markdown

# Use Ollama for local testing (requires ollama running with gemma3:1b-it-qat model)
# Features Gemma-specific function calling with tool_code blocks
uv run python -m smolval.cli eval prompts/simple_test.txt -c config/ollama.yaml --format markdown

# Use Google Gemini (requires llm-gemini plugin and GEMINI_API_KEY)
uv run python -m smolval.cli eval prompts/simple_test.txt -c config/example-gemini.yaml --format markdown

# Use Claude Code agent (requires claude CLI installed and ANTHROPIC_API_KEY)
uv run python -m smolval.cli eval prompts/simple_test.txt -c config/example-claude-code.yaml --format markdown
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

# Google Gemini:
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  -e GEMINI_API_KEY \
  smolval eval prompts/simple_test.txt -c config/example-gemini.yaml --format markdown

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

# Run by test markers (defined in pyproject.toml)
uv run pytest -m "not integration and not slow"  # Fast unit tests only
uv run pytest -m integration                     # Integration tests only
uv run pytest -m "requires_docker"               # Docker-dependent tests
uv run pytest -m "requires_api_keys"             # Tests requiring real API keys

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

# MANDATORY: Pre-commit quality checks (run before every commit)
uv run black src/ tests/ && uv run isort src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/
```

### Pre-Commit Workflow
**IMPORTANT**: Always run the complete quality check pipeline before committing:

1. **Format code**: `uv run black src/ tests/`
2. **Sort imports**: `uv run isort src/ tests/`
3. **Check linting**: `uv run ruff check src/ tests/`
4. **Type check**: `uv run mypy src/`
5. **Run tests**: `uv run pytest`
6. **Then commit and push**

This ensures consistent code quality and prevents formatting issues in commits.

## Core Architecture

### Component Relationships
1. **CLI** (`cli.py`) → **Config** (`config.py`) → **Agent** (`agent.py`) workflow
2. **Agent** orchestrates **LLMClient** + **MCPClientManager** in ReAct loop (ReAct agent)
3. **ClaudeCodeAgent** uses Claude Code CLI subprocess with MCP server management (Claude Code agent)
4. **MCPClientManager** (`mcp_client.py`) discovers tools from multiple MCP servers, presents unified interface
5. **LLMClient** (`llm_client.py`) provides unified interface for Anthropic Claude, OpenAI, Google Gemini, and Ollama models with provider-specific function calling
6. **ResultsFormatter** (`results.py`) handles multi-format output with Jinja2 templating

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
- **Default Config**: `config/example-anthropic.yaml` with filesystem, fetch servers (Claude 4 Sonnet)
- **Alternative Configs**: 
  - `config/example-openai.yaml` (GPT-4o models)
  - `config/example-gemini.yaml` (Google Gemini models) 
  - `config/example-ollama.yaml` (Ollama with gemma3:1b-it-qat)
  - `config/example-claude-code.yaml` (Claude Code agent)
- **Required Environment**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` (not required for Ollama)
- **LLM Library**: Uses `datasette llm` with automatic plugin loading for providers
- **Plugin Requirements**: Install `llm-gemini` for Google Gemini support

### MCP Server Support
- **NPM-based servers**: `@modelcontextprotocol/server-filesystem`
- **Python-based servers**: `mcp-server-fetch` (web content retrieval via uvx)
- **Docker-based servers**: `mcp/sqlite` (requires Docker socket access)
- **Mixed deployment**: Can run NPM, Python, and Docker-based servers simultaneously

## Important Implementation Details

### Agent Execution Flow

#### ReAct Agent
The ReAct agent runs a ReAct loop with:
1. **Thought**: LLM reasoning about next action
2. **Action**: Tool call with arguments
3. **Observation**: Tool execution result
4. Continues until task completion or max iterations (15)

#### Claude Code Agent
The Claude Code agent uses Claude Code CLI:
1. **Setup**: Configures MCP servers using `claude mcp add` commands
2. **Permissions**: Uses `--allowedTools` and `--disallowedTools` flags for secure development permissions (file creation, npm, git)
3. **Execution**: Runs `claude -p "prompt" --output-format stream-json --allowedTools [...] --disallowedTools [...]`
4. **Parsing**: Processes streaming JSON output into AgentResult format
5. **Cleanup**: Removes temporary MCP server configurations

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