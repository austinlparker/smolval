# Getting Started with smolval

This guide will help you get up and running with smolval for evaluating MCP (Model Context Protocol) servers using our containerized approach with Claude Code CLI.

## Prerequisites

- Docker
- ANTHROPIC_API_KEY

## Installation

```bash
# Clone the repository
git clone https://github.com/austinlparker/smolval
cd smolval

# Build the container (includes Claude Code CLI and all tools)
docker build -t ghcr.io/austinlparker/smolval .

# Verify installation
docker run --rm ghcr.io/austinlparker/smolval --help
```

## Quick Start


#### 1. Set up your API key

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Or create a .env file in project directory
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

#### 2. Run your first evaluation

```bash
# Basic evaluation using Claude Code's built-in tools
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt

# With specific output format
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --format html

# With verbose output (see Claude CLI interaction)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/simple_test.txt --verbose
```

#### 3. For MCP servers requiring Docker

```bash
# Enable Docker-in-Docker for containerized MCP servers
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/database-test.txt
```


## Basic Usage

#### Single Evaluation

```bash
# Evaluate a single prompt
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/your-prompt.txt

# With custom timeout
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/your-prompt.txt --timeout 600
```

#### MCP Configuration

```bash
# Use custom .mcp.json configuration
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/test.txt --mcp-config /workspace/.mcp.json
```


## Configuration

### MCP Server Configuration

smolval uses the standard `.mcp.json` configuration format:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {}
    },
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    }
  }
}
```

**Note**: Claude Code includes built-in filesystem and web fetch capabilities, so MCP servers are only needed for additional functionality.

### Container Path Considerations

When using the container, ensure paths are relative to the mounted workspace:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {}
    }
  }
}
```

### Common MCP Server Examples

#### Filesystem Server (NPM)
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {}
    }
  }
}
```

#### SQLite Server (Docker)
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

#### GitHub Integration
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

## Output Formats

smolval supports multiple output formats:

### Container Usage

```bash
# JSON (Default)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/example.txt --format json

# Markdown
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/example.txt --format markdown

# HTML
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/example.txt --format html

# CSV
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/example.txt --format csv
```


## Writing Your First Prompt

Create a simple prompt file:

```
# prompts/my-test.txt
Test basic MCP server functionality:

1. List files in the current directory
2. Read the contents of README.md if it exists
3. Create a temporary file with test content
4. Verify the file was created successfully

Success criteria:
- All file operations complete without errors
- File contents are accurately read and written
- Temporary files are properly managed
```

## Common Use Cases

### Testing File Operations
```bash
# Test Claude Code's built-in filesystem capabilities
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/file-operations.txt
```

### Web Content Analysis
```bash
# Test Claude Code's built-in web fetch capabilities
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/web-analysis.txt
```

### Database Operations
```bash
# Test with SQLite MCP server
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/database-test.txt --mcp-config /workspace/.mcp.json
```

### Custom MCP Server Testing
```bash
# Test your own MCP server configurations
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval eval /workspace/prompts/custom-test.txt --mcp-config /workspace/.mcp.json
```

## Troubleshooting

### Container Issues

1. **Missing API Key**
   ```
   Error: No API key found
   ```
   Solution: Ensure `-e ANTHROPIC_API_KEY` is set in docker run command

2. **Volume Mount Issues**
   ```
   Error: Cannot access /workspace/prompts
   ```
   Solution: Ensure correct volume mount `-v $(pwd):/workspace`

3. **Docker-in-Docker Issues**
   ```
   Error: Cannot connect to Docker daemon
   ```
   Solution: Add Docker socket mount `-v /var/run/docker.sock:/var/run/docker.sock`

4. **Claude CLI Issues**
   ```
   Error: Claude CLI not found
   ```
   Solution: Rebuild container - Claude CLI is pre-installed in the image

5. **MCP Server Path Issues**
   ```
   Error: MCP server command failed
   ```
   Solution: Use `/workspace` paths in .mcp.json for container compatibility


### Getting Help

- Check the [Configuration Reference](config-reference.md) for MCP configuration details
- See [Writing Prompts](writing-prompts.md) for effective prompt guidelines
- Review [Architecture](architecture.md) for technical implementation details
- Check container logs: `docker logs <container-id>`
- Open an issue on GitHub for bugs or feature requests

## Next Steps

- Learn about [Writing Effective Prompts](writing-prompts.md)
- Explore [CLI Reference](cli-reference.md) for all command options
- Check out the [Examples](examples/) directory for sample configurations
- Set up automated evaluation pipelines with CI/CD using the container

## Development Workflow

### Interactive Development

```bash
# Start an interactive container for development
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  -w /workspace \
  ghcr.io/austinlparker/smolval:dev bash

# Inside the container, you have access to:
# - claude --version
# - npm, npx, node
# - docker
# - git, vim, tree, jq
# - uv, uvx
# - All smolval commands
```

### Docker Compose (Optional)

For easier development setup, create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  smolval:
    build: .
    volumes:
      - .:/workspace
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - ANTHROPIC_API_KEY
    working_dir: /workspace
    command: tail -f /dev/null
```

Then run: `docker-compose up -d && docker-compose exec smolval bash`

This should get you started with smolval's container-first approach! For more advanced usage and customization, continue reading the other documentation sections.