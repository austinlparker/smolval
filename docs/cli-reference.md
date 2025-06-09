# CLI Reference

This document provides a complete reference for the smolval command-line interface. smolval is designed to run in a Docker container with Claude Code CLI pre-installed.

## Installation

### Container Usage (Recommended)

```bash
# Build the container
docker build -t ghcr.io/austinlparker/smolval .

# Run help to verify installation
docker run --rm ghcr.io/austinlparker/smolval --help
```


## Main Command

```
smolval [OPTIONS] COMMAND [ARGS]...
```

A lightweight MCP server evaluation agent for testing Model Context Protocol implementations.

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--verbose, -v` | Enable verbose logging (INFO level) |
| `--debug` | Enable debug logging (DEBUG level) |
| `--no-banner` | Disable ASCII banner display |
| `--help` | Show help message and exit |

### Container Examples
```bash
# Show version
docker run --rm ghcr.io/austinlparker/smolval --version

# Run with verbose logging
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval --verbose eval /workspace/prompts/test.txt

# Run with debug logging and no banner
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval --debug --no-banner eval /workspace/prompts/test.txt
```


## Commands

**Note**: The current version focuses on single prompt evaluation. Batch processing and server comparison features have been simplified to focus on Claude Code CLI integration.

### eval

Evaluate MCP servers using Claude Code CLI with a single prompt file.

```
smolval eval PROMPT_FILE [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `PROMPT_FILE` | Path | Yes | Path to the prompt file to evaluate |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mcp-config` | `.mcp.json` | MCP configuration file path |
| `--timeout` | `300` | Timeout in seconds for Claude Code execution |
| `--format` | `json` | Output format: json, markdown, html, csv |
| `--output` | _(auto-generated)_ | Output file path |
| `--verbose` | `False` | Show Claude CLI stdout/stderr |
| `--no-progress` | `False` | Disable progress indicator |

#### Container Examples

```bash
# Basic evaluation using Claude Code's built-in tools
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/file-operations.txt

# With custom MCP configuration
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/database-test.txt --mcp-config /workspace/.mcp.json

# With specific output format
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/test.txt --format html --output /workspace/report.html

# Verbose output to see Claude CLI interaction
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/test.txt --verbose

# With custom timeout
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/test.txt --timeout 600
```


#### Output

Generates a single result file in the specified format:
- **JSON**: Structured evaluation data with steps, metadata, and results
- **Markdown**: Human-readable report with conversation flow
- **HTML**: Rich HTML report with syntax highlighting and formatting
- **CSV**: Tabular data suitable for analysis and comparison

Default output filename: `smolval_result_{timestamp}.{format}`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for Claude Code CLI |
| `CLAUDE_CONFIG_DIR` | No | Claude CLI configuration directory (defaults to `/app/.claude` in container) |

## MCP Configuration

smolval uses the standard `.mcp.json` configuration format:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {}
    }
  }
}
```

**Note**: Claude Code has filesystem and web fetch capabilities built-in, so MCP servers are only needed for additional functionality.

## Docker Usage Patterns

### Volume Mounting

```bash
# Basic workspace mount
-v $(pwd):/workspace

# Docker-in-Docker for containerized MCP servers
-v /var/run/docker.sock:/var/run/docker.sock

# Custom output directory
-v $(pwd)/results:/results

# Mount specific config file
-v $(pwd)/custom.mcp.json:/app/.mcp.json
```

### Common Docker Commands

```bash
# Development shell
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  -w /workspace \
  ghcr.io/austinlparker/smolval bash

# Run tests
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval uv run pytest

# Code quality checks
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval bash -c "uv run black src/ tests/ && uv run mypy src/"
```

## Docker Compose Usage

For easier development, use the included `docker-compose.yml`:

```bash
# Start development container
docker-compose up -d

# Interactive development shell
docker-compose exec smolval bash

# Run evaluation
docker-compose exec smolval smolval eval /workspace/prompts/test.txt

# Stop and clean up
docker-compose down
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: No API key found
   ```
   Solution: Ensure `-e ANTHROPIC_API_KEY` is included in docker run

2. **File Not Found**
   ```
   Error: No such file or directory: '/workspace/prompts/test.txt'
   ```
   Solution: Check volume mount and file paths

3. **MCP Server Issues**
   ```
   Error: MCP server command failed
   ```
   Solution: For Docker-based MCP servers, add Docker socket mount

4. **Permission Issues**
   ```
   Error: Permission denied
   ```
   Solution: Check file permissions or run as root with `--user root`

### Debug Commands

```bash
# Check Claude CLI version in container
docker run --rm ghcr.io/austinlparker/smolval claude --version

# List available tools
docker run --rm ghcr.io/austinlparker/smolval which claude

# Check environment
docker run --rm -e ANTHROPIC_API_KEY ghcr.io/austinlparker/smolval env | grep ANTHROPIC

# Test MCP config parsing
docker run --rm \
  -v $(pwd):/workspace \
  ghcr.io/austinlparker/smolval cat /workspace/.mcp.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Missing required arguments |
| 3 | Configuration error |
| 4 | Claude CLI not found |
| 5 | Timeout error |