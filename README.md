# smolval

A lightweight MCP server evaluation agent powered by LLM-driven ReAct.

## Features

- Evaluate MCP servers with LLM-guided reasoning and tooling.
- Support for multiple LLM providers: Anthropic Claude, OpenAI, Google Gemini, and Ollama.
- Batch evaluations and server comparisons.
- Cross-provider model comparisons.
- Outputs results in JSON, CSV, Markdown, and HTML.

## Installation

Install from PyPI:
```bash
pip install smolval
```

Or install from source:
```bash
git clone https://github.com/your-repo/smolval.git
cd smolval
pip install .
```

## Quickstart

1. Configure MCP servers and LLM in `config/example-*.yaml`.
2. Prepare a prompt file (`.txt` or `.md`) in `prompts/`.
3. Run evaluation:
```bash
smolval eval prompts/example.txt --config config/example-anthropic.yaml --output-dir results
```
4. View results in the output directory.

## Commands

- `smolval eval`: Evaluate a single prompt.
- `smolval batch`: Run batch evaluations over a directory of prompts.
- `smolval compare`: Compare two MCP servers across prompts.
- `smolval compare-providers`: Compare different LLM providers with the same prompts.

Run `smolval --help` for detailed options.

## Configuration

Sample configs:
```text
config/example-anthropic.yaml
config/example-openai.yaml
config/example-gemini.yaml
config/example-ollama.yaml
```

For Google Gemini, install the LLM plugin:
```bash
pip install llm-gemini
```

## Prompts

Example prompts are in the `prompts/` folder. Create your own to customize evaluations.

## Design & Testing

See [DESIGN.md](DESIGN.md) and [CLAUDE.md](CLAUDE.md) for architecture notes.
Refer to [tests/README.md](tests/README.md) for running unit and integration tests.

## Testing & Coverage

For a detailed testing guide and markers, see [tests/README.md](tests/README.md). To generate a code coverage report:

```bash
uv run pytest --cov=smolval --cov-report=html
```