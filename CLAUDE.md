# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

smolval is a lightweight, containerized Python wrapper around Claude Code CLI for evaluating MCP servers. It provides a self-contained Docker environment with Claude Code CLI, development tools, and MCP server support built-in. Uses the standard `.mcp.json` configuration format and focuses solely on Claude Code agent execution.

## Key Features

- **Self-Contained Container**: Claude Code CLI and all tools pre-installed, no host dependencies
- **Progress Indicators**: Visual feedback during Claude Code execution with elapsed time tracking
- **Multiple Output Formats**: JSON, CSV, Markdown, and HTML output options  
- **Verbose Mode**: Detailed logging with Claude CLI stdout/stderr output
- **Environment Integration**: Automatic `.env` file loading for API keys
- **Standard Configuration**: Uses `.mcp.json` format compatible with Claude Desktop/Cursor
- **Docker-in-Docker Support**: Full MCP server isolation with container support

## Development Commands

### Prerequisites
- Docker
- ANTHROPIC_API_KEY environment variable

### Container-First Development (Recommended)
```bash
# Build development image
docker build -t ghcr.io/austinlparker/smolval:dev .

# Interactive development container
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash

# Inside container, all tools are available:
# uv sync, claude --version, npm, docker, git, etc.
```


### Running the Application

#### Container Usage (Recommended)
```bash
# Build the Docker image
docker build -t ghcr.io/austinlparker/smolval .

# Basic evaluation (no MCP servers - uses Claude Code's built-in tools)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt

# With MCP servers requiring Docker-in-Docker
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --mcp-config /workspace/.mcp.json

# Different output formats
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --format json --output /workspace/results.json

# Verbose output (shows Claude CLI output)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --verbose

# Custom timeout
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --timeout 600
```


### Testing

#### Container Testing (Recommended)
```bash
# Run tests in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest

# Run with coverage
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest --cov=smolval

# Run specific test file
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest tests/test_claude_code_agent.py
```


### Code Quality

#### Container Quality Checks (Recommended)
```bash
# All quality checks in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash -c "uv run black src/ tests/ && uv run isort src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/"

# Individual checks
docker run --rm -v $(pwd):/workspace -w /workspace ghcr.io/austinlparker/smolval:dev uv run mypy src/
docker run --rm -v $(pwd):/workspace -w /workspace ghcr.io/austinlparker/smolval:dev uv run black src/ tests/
docker run --rm -v $(pwd):/workspace -w /workspace ghcr.io/austinlparker/smolval:dev uv run ruff check src/ tests/
```


### Pre-Commit Workflow
**IMPORTANT**: Always run the complete quality check pipeline before committing:

#### Container Workflow (Recommended)
```bash
# One-command quality check in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash -c "uv run black src/ tests/ && uv run isort src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/ && uv run pytest"

# Then commit and push
git add -A && git commit -m "your message" && git push
```


## Core Architecture

### Simplified Design
1. **CLI** (`cli.py`) → **ClaudeCodeAgent** (`agent.py`) → **Claude Code CLI subprocess**
2. **MCP Configuration** via standard `.mcp.json` files (Claude Code handles MCP server management)
3. **Output Formatting** (`results.py`, `output_manager.py`) for multiple formats

### Key Components
- **ClaudeCodeAgent**: Executes Claude Code CLI as subprocess with optional MCP server config
- **OutputManager**: Formats results as JSON, Markdown, CSV, or HTML
- **CLI**: Simple interface with single `eval` command

### MCP Configuration
Uses standard `.mcp.json` format compatible with Claude Desktop, Cursor, and other MCP-aware tools.

**Note**: Claude Code has filesystem and web fetch capabilities built-in, so MCP servers are only needed for additional functionality like databases, specialized APIs, etc.

Example for SQLite database access:
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    }
  }
}
```

### Dependencies
Minimal dependency set:
- `pydantic` - Data validation and models
- `click` - CLI interface
- `jinja2` - Output templating
- `markdown-it-py` - Markdown processing

### MCP Server Support
Claude Code includes built-in support for:
- **Filesystem operations** - Built into Claude Code
- **Web content fetching** - Built into Claude Code

Additional MCP servers can be configured for:
- **Database access**: SQLite, PostgreSQL, etc.
- **API integrations**: GitHub, Slack, etc.
- **Specialized tools**: Custom business logic, etc.

## Usage Examples

### No MCP Configuration (Default)
Claude Code works out of the box with filesystem and web fetch capabilities:
```bash
# Uses Claude Code's built-in tools only
uv run python -m smolval.cli eval prompts/simple_test.txt
```

### With Additional MCP Servers

**SQLite database access:**
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    }
  }
}
```

**GitHub integration:**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Requirements

#### Container Usage (Recommended)
- **Docker**: Only requirement for host system
- **ANTHROPIC_API_KEY**: Required environment variable
- **MCP Servers**: Pre-installed in container or via Docker-in-Docker


## Important Implementation Details

### Agent Execution
1. **Setup**: Uses Claude Code CLI from container (`/usr/local/bin/claude`)
2. **Configuration**: Uses `.mcp.json` from specified path or current directory (optional)
3. **Execution**: Runs `claude -p "prompt" --output-format stream-json --verbose`
4. **Parsing**: Processes streaming JSON output into structured results
5. **Output**: Formats results using Jinja2 templates

### Error Handling
- Timeout support with configurable limits
- Graceful handling of missing Claude CLI
- Clear error messages for configuration issues
- Works without MCP configuration (uses Claude Code's built-in tools)

### Output Formats
Results can be formatted as JSON, CSV, Markdown, or HTML using templates in the `templates/` directory.

## Migration Notes

This version has been dramatically simplified from the original multi-provider ReAct agent system:
- **Removed**: LLM client abstractions, MCP client management, judge functionality, comparison features
- **Added**: Standard `.mcp.json` support, focus on Claude Code CLI integration
- **Simplified**: Single agent type, minimal configuration, reduced dependencies

The tool is now positioned as a lightweight, containerized wrapper around Claude Code CLI for systematic MCP server evaluation, leveraging Claude Code's built-in filesystem and web capabilities while allowing additional MCP servers as needed.

## Container Architecture Benefits

- **Zero Host Dependencies**: Only requires Docker and API key
- **Consistent Environment**: Same Claude CLI version across all environments  
- **Isolation**: No configuration conflicts with host Claude installations
- **CI/CD Ready**: Perfect for automated evaluation pipelines
- **Easy Distribution**: Single container includes everything needed