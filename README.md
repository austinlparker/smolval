# smolval

A lightweight, containerized Python application for evaluating MCP (Model Context Protocol) servers using Claude Code CLI. smolval provides a self-contained Docker environment with Claude Code CLI, development tools, and MCP server support built-in for systematic MCP server evaluation.

## ✨ Features

- **Self-Contained Container**: Claude Code CLI and all tools pre-installed, zero host dependencies
- **MCP Server Evaluation**: Systematic testing using Claude Code's agent capabilities
- **Docker-in-Docker Support**: Full MCP server isolation with container support
- **Multiple Output Formats**: JSON, CSV, Markdown, and HTML results
- **Progress Indicators**: Visual feedback during evaluation with elapsed time tracking
- **Standard Configuration**: Uses `.mcp.json` format compatible with Claude Desktop/Cursor

## 🚀 Quick Start

### Prerequisites

- Docker
- ANTHROPIC_API_KEY

### One-Command Setup

```bash
# Clone repository
git clone https://github.com/austinlparker/smolval.git
cd smolval

# Build container (includes Claude Code CLI and all tools)
docker build -t ghcr.io/austinlparker/smolval .

# Run your first evaluation - no local installation required!
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt
```

### For MCP Servers Requiring Docker

```bash
# Enable Docker-in-Docker for containerized MCP servers
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/database-test.txt
```

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started.md)** - Installation and setup guide
- **[CLI Reference](docs/cli-reference.md)** - Complete command-line documentation
- **[Configuration](docs/config-reference.md)** - Configuration options and examples
- **[Writing Prompts](docs/writing-prompts.md)** - Guide to creating effective evaluation prompts
- **[Examples](docs/examples/)** - Sample prompts and configurations
- **[Architecture](docs/architecture.md)** - Technical design and implementation details

## 🛠️ Commands

### Single Evaluation
```bash
# Basic evaluation using Claude Code's built-in tools
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/file-test.txt

# With specific output format
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/file-test.txt --format html
```

### Custom MCP Configuration
```bash
# Use custom .mcp.json configuration
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/test.txt --mcp-config /workspace/.mcp.json
```

Run `docker run --rm ghcr.io/austinlparker/smolval --help` for all options.

## ⚙️ Configuration

smolval uses the standard `.mcp.json` configuration format compatible with Claude Desktop and Cursor:

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

**Note**: Claude Code has filesystem and web fetch capabilities built-in, so MCP servers are only needed for additional functionality like databases, APIs, etc.

Example configurations are available in the [`docs/examples/`](docs/examples/) directory.

## 🐳 Container Features

### Pre-installed Tools
- **Claude Code CLI**: Latest version ready to use
- **Node.js & npm/npx**: For NPM-based MCP servers  
- **Docker CLI**: For containerized MCP servers
- **Development Tools**: git, vim, tree, jq, uvx
- **MCP Servers**: Common servers pre-installed for faster startup

### Volume Mounting Strategy
```bash
# Basic workspace mount
-v $(pwd):/workspace

# Docker-in-Docker support
-v /var/run/docker.sock:/var/run/docker.sock

# Custom output directory
-v $(pwd)/results:/results
```

### Environment Variables
```bash
# Required
-e ANTHROPIC_API_KEY

# Optional
-e CLAUDE_CONFIG_DIR=/app/.claude
```

## 🧪 Testing

Run the test suite in container:
```bash
# Build development image
docker build -t ghcr.io/austinlparker/smolval:dev .

# Unit tests only
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest -m "not integration and not slow"

# All tests
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest

# With coverage
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev uv run pytest --cov=smolval --cov-report=html
```

See [`tests/README.md`](tests/README.md) for detailed testing information.

## 📊 MCP Server Support

smolval supports various MCP server types through the container:

- **Built-in Claude Code Tools**: Filesystem operations, web content fetching
- **Pre-installed NPM**: `@modelcontextprotocol/server-filesystem`, `@modelcontextprotocol/server-memory`
- **Docker-based**: `mcp/sqlite`, custom containers via Docker-in-Docker
- **Python-based**: Any uvx-installable MCP servers

## 🔧 Development

### Container-First Development

```bash
# Interactive development container
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash

# Code quality checks in container
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash -c "uv run black src/ tests/ && uv run isort src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/"
```


### Project Structure

```
smolval/
├── src/smolval/          # Main application code
├── docs/                 # Documentation
├── tests/                # Test suite
├── prompts/              # Example evaluation prompts
└── results/              # Generated evaluation results
```

## 📋 Requirements

- **Docker**: Only host system requirement
- **ANTHROPIC_API_KEY**: Environment variable for Claude Code CLI

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read the [Architecture documentation](docs/architecture.md) and check the test suite before submitting changes.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the quality checks
5. Submit a pull request

## 📚 Learn More

- [Model Context Protocol](https://modelcontextprotocol.io/) - Learn about MCP
- [ReAct Pattern](https://arxiv.org/abs/2210.03629) - The reasoning pattern used by smolval
- [Project Documentation](docs/) - Comprehensive guides and references