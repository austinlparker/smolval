# Getting Started with smolval

This guide will help you get up and running with smolval for evaluating MCP (Model Context Protocol) servers.

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ (for NPM-based MCP servers)
- Docker (optional, for containerized MCP servers)
- API keys for your chosen LLM provider

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/austinlparker/smolval
cd smolval

# Install dependencies
uv sync --all-extras
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/austinlparker/smolval
cd smolval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Set up your API key

```bash
# For Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="your-api-key-here"

# Or for OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

### 2. Install MCP servers

Install the basic MCP servers for testing:

```bash
# Install filesystem server (NPM)
npm install -g @modelcontextprotocol/server-filesystem

# Install web fetch server (Python)
uv tool install mcp-server-fetch

# Pull SQLite server (Docker)
docker pull mcp/sqlite
```

### 3. Run your first evaluation

```bash
# Run a simple evaluation
uv run python -m smolval.cli eval prompts/example.txt

# Run with specific output format
uv run python -m smolval.cli eval prompts/example.txt --format markdown

# Run with custom configuration
uv run python -m smolval.cli eval prompts/example.txt -c config/example-anthropic.yaml
```

## Basic Usage

### Single Evaluation

Evaluate a single prompt against the configured MCP servers:

```bash
uv run python -m smolval.cli eval prompts/your-prompt.txt
```

### Batch Evaluation

Run multiple prompts from a directory:

```bash
uv run python -m smolval.cli batch prompts/ -o results/
```

### Server Comparison

Compare how different MCP servers handle the same prompt:

```bash
uv run python -m smolval.cli compare \
  --baseline filesystem \
  --test fetch \
  prompts/example.txt \
  --format html
```

## Configuration

### Basic Configuration

Create a custom configuration file:

```yaml
# config/my-config.yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env: {}

llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1

evaluation:
  timeout_seconds: 120
  max_iterations: 15
  output_format: "json"
```

### Using Different LLM Providers

#### Anthropic Claude (Default)
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
```

#### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
```

#### Ollama (Local)
```yaml
llm:
  provider: "ollama"
  model: "gemma2:2b"
  base_url: "http://localhost:11434"
```

### MCP Server Configuration

#### Filesystem Server
```yaml
- name: "filesystem"
  command: ["npx", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
  env: {}
```

#### Web Fetch Server
```yaml
- name: "fetch"
  command: ["uvx", "mcp-server-fetch"]
  env: {}
```

#### SQLite Server (Docker)
```yaml
- name: "sqlite"
  command: ["docker", "run", "--rm", "-i", "-v", "data:/data", "mcp/sqlite", "--db-path", "/data/test.db"]
  env: {}
```

## Output Formats

smolval supports multiple output formats:

### JSON (Default)
```bash
uv run python -m smolval.cli eval prompts/example.txt --format json
```

### Markdown
```bash
uv run python -m smolval.cli eval prompts/example.txt --format markdown
```

### HTML
```bash
uv run python -m smolval.cli eval prompts/example.txt --format html
```

### CSV
```bash
uv run python -m smolval.cli eval prompts/example.txt --format csv
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
# Test filesystem server capabilities
uv run python -m smolval.cli eval prompts/file-operations.txt -c config/filesystem-only.yaml
```

### Web Content Analysis
```bash
# Test web fetching capabilities
uv run python -m smolval.cli eval prompts/web-analysis.txt -c config/fetch-only.yaml
```

### Database Queries
```bash
# Test database operations
uv run python -m smolval.cli eval prompts/database-test.txt -c config/sqlite-only.yaml
```

### Performance Benchmarking
```bash
# Compare server performance
uv run python -m smolval.cli compare \
  --baseline filesystem \
  --test filesystem-v2 \
  prompts/performance-test.txt \
  --format html -o results/benchmark.html
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: No API key found for provider 'anthropic'
   ```
   Solution: Set your API key in environment variables or .env file

2. **MCP Server Not Found**
   ```
   Error: Command not found: npx
   ```
   Solution: Install Node.js and the required MCP servers

3. **Permission Denied**
   ```
   Error: Permission denied accessing /path
   ```
   Solution: Check file permissions or use a different directory

4. **Docker Issues**
   ```
   Error: Cannot connect to Docker daemon
   ```
   Solution: Start Docker service or remove Docker-based servers from config

### Getting Help

- Check the [Configuration Reference](config-reference.md) for detailed options
- See [Writing Prompts](writing-prompts.md) for prompt guidelines
- Review [Architecture](architecture.md) for technical details
- Open an issue on GitHub for bugs or feature requests

## Next Steps

- Learn about [Writing Effective Prompts](writing-prompts.md)
- Explore [Advanced Configuration](config-reference.md)
- Check out the [Examples](examples/) directory
- Set up [CI/CD Integration](ci-cd.md) for automated testing

## Docker Usage

For containerized deployments:

```bash
# Build the image
docker build -t smolval .

# Run with API key
docker run --rm \
  -e ANTHROPIC_API_KEY \
  smolval eval prompts/example.txt --format markdown

# Run with Docker socket (for SQLite server)
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  -e ANTHROPIC_API_KEY \
  smolval eval prompts/example.txt
```

This should get you started with smolval! For more advanced usage and customization, continue reading the other documentation sections.