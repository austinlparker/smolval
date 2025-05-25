# smolval

A lightweight MCP server evaluation agent for testing and validating Model Context Protocol implementations.

## Features

✅ **Complete Implementation**:
- 🤖 ReAct agent loop with LLM integration (Anthropic Claude & OpenAI)
- 🔧 MCP client manager supporting multiple concurrent servers
- 📊 Comprehensive evaluation framework with step-by-step tracking
- 📈 Advanced comparison and batch evaluation capabilities
- 📝 Multiple output formats (JSON, CSV, Markdown, HTML)
- 🧪 Extensive test suite (41 tests passing)

## Quick Start

### Using Docker (Recommended)

```bash
# Build the container
docker build -t smolval .

# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run an evaluation
docker run -e ANTHROPIC_API_KEY -v $(pwd)/prompts:/app/prompts -v $(pwd)/config:/app/config \
  smolval eval prompts/simple_test.txt --format markdown
```

### Local Development

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run tests
uv run pytest

# Run an evaluation
uv run python -m smolval.cli eval prompts/simple_test.txt --format markdown
```

## Configuration

The default configuration uses MCP servers that don't require API keys:

```yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env: {}
  - name: "memory"
    command: ["npx", "@modelcontextprotocol/server-memory"]
    env: {}
  - name: "sqlite"
    command: ["npx", "@modelcontextprotocol/server-sqlite", "./data/example.db"]
    env: {}

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1

evaluation:
  timeout_seconds: 120
  max_iterations: 15
  output_format: "json"
```

## Usage

### Single Evaluation

```bash
# Basic evaluation with console output
uv run python -m smolval.cli eval prompts/simple_test.txt

# Save as JSON
uv run python -m smolval.cli eval prompts/file_operations.txt -o results.json

# Generate markdown report
uv run python -m smolval.cli eval prompts/simple_test.txt --format markdown -o report.md

# Generate HTML report
uv run python -m smolval.cli eval prompts/file_operations.txt --format html -o report.html
```

### Batch Evaluation

```bash
# Run all prompts in directory
uv run python -m smolval.cli batch prompts/ -o results/ --format json

# Filter to specific servers
uv run python -m smolval.cli batch prompts/ --servers filesystem,memory --format csv

# Generate comprehensive markdown reports
uv run python -m smolval.cli batch prompts/ -o batch_results/ --format markdown
```

### Server Comparison

```bash
# Compare two servers across all prompts
uv run python -m smolval.cli compare --baseline filesystem --test memory prompts/

# Save comparison as CSV for analysis
uv run python -m smolval.cli compare --baseline filesystem --test memory prompts/ \
  --format csv -o comparison.csv

# Generate detailed markdown comparison report
uv run python -m smolval.cli compare --baseline filesystem --test memory prompts/ \
  --format markdown -o comparison_report.md
```

### Output Formats

- **JSON**: Structured data for programmatic analysis
- **CSV**: Spreadsheet-compatible for data analysis
- **Markdown**: Human-readable reports with detailed step traces
- **HTML**: Formatted web-ready reports

## Writing Evaluation Prompts

Create `.txt` files in the `prompts/` directory with clear instructions and success criteria:

```
Please use the available MCP tools to:

1. List files in the current directory
2. Read the contents of any .txt files found
3. Summarize what you discovered

Expected outcome: Successfully demonstrate file system operations.

Success criteria:
- File listing completed without errors
- At least one .txt file read successfully
- Operations completed within 30 seconds
```

## Development

This project uses Test-Driven Development (TDD). Write tests first:

```bash
# Run tests with coverage
uv run pytest --cov=smolval

# Type checking
uv run mypy src/

# Code formatting
uv run black src/ tests/
uv run isort src/ tests/

# Linting
uv run ruff check src/ tests/
```

## Architecture

See [DESIGN.md](DESIGN.md) for detailed architecture and design decisions.