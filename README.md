# smolval

A lightweight Python application for evaluating MCP (Model Context Protocol) servers using LLM agents. smolval implements a ReAct (Reason + Act) pattern to systematically test MCP server implementations through structured evaluation prompts.

## ✨ Features

- **LLM-Driven Evaluation**: Uses ReAct pattern with LLM agents to test MCP servers
- **Multiple LLM Providers**: Supports Anthropic Claude, OpenAI, and Ollama
- **Batch Processing**: Evaluate multiple prompts and compare different servers
- **Rich Output Formats**: Results in JSON, CSV, Markdown, and HTML
- **Docker Support**: Containerized deployment with MCP server compatibility
- **Comprehensive Testing**: Built-in test suite with high coverage

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 18+ (for NPM-based MCP servers)
- Docker (optional, for containerized MCP servers)

### Installation

```bash
git clone https://github.com/austinlparker/smolval.git
cd smolval
uv sync --all-extras
```

### Basic Usage

1. **Set up your API key:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
# or for OpenAI: export OPENAI_API_KEY="your-api-key-here"
```

2. **Install MCP servers:**
```bash
npm install -g @modelcontextprotocol/server-filesystem
uv tool install mcp-server-fetch
```

3. **Run your first evaluation:**
```bash
uv run python -m smolval.cli eval prompts/example.txt
```

4. **View results in the generated output directory**

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
uv run python -m smolval.cli eval prompts/file-test.txt
```

### Batch Evaluation
```bash
uv run python -m smolval.cli batch prompts/ --output-dir results/
```

### Server Comparison
```bash
uv run python -m smolval.cli compare \
  --baseline filesystem \
  --test fetch \
  prompts/ \
  --format html
```

Run `uv run python -m smolval.cli --help` for all options.

## ⚙️ Configuration

smolval uses YAML configuration files with support for environment variable expansion:

```yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "."]
    env: {}

llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1

evaluation:
  timeout_seconds: 120
  max_iterations: 15
  output_format: "markdown"
```

Example configurations are available in [`docs/examples/`](docs/examples/).

## 🐳 Docker Usage

```bash
# Build the image
docker build -t smolval .

# Run evaluation
docker run --rm \
  -v $(pwd)/prompts:/prompts \
  -v $(pwd)/results:/results \
  -e ANTHROPIC_API_KEY \
  smolval eval /prompts/example.txt --output-dir /results
```

For MCP servers requiring Docker (like SQLite), mount the Docker socket:
```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  -e ANTHROPIC_API_KEY \
  smolval eval prompts/database-test.txt
```

## 🧪 Testing

Run the test suite:
```bash
# Unit tests only
uv run pytest -m "not integration and not slow"

# All tests
uv run pytest

# With coverage
uv run pytest --cov=smolval --cov-report=html
```

See [`tests/README.md`](tests/README.md) for detailed testing information.

## 📊 MCP Server Support

smolval supports various MCP server types:

- **NPM-based**: `@modelcontextprotocol/server-filesystem`
- **Python-based**: `mcp-server-fetch` (web content retrieval)
- **Docker-based**: `mcp/sqlite` (database operations)

## 🔧 Development

### Code Quality

```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Run all quality checks
uv run black src/ tests/ && uv run isort src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/
```

### Project Structure

```
smolval/
├── src/smolval/          # Main application code
├── docs/                 # Documentation
├── tests/                # Test suite
├── config/               # Example configurations
├── prompts/              # Example evaluation prompts
└── results/              # Generated evaluation results
```

## 📋 Requirements

- **Python**: 3.11+
- **Core Dependencies**: `mcp`, `llm`, `pydantic`, `click`, `pyyaml`, `rich`, `jinja2`
- **Development**: `pytest`, `black`, `isort`, `mypy`, `ruff`

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