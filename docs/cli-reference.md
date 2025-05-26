# CLI Reference

This document provides a complete reference for the smolval command-line interface.

## Installation

First, ensure smolval is installed and available in your PATH:

```bash
# With uv (recommended)
uv sync --all-extras

# Run commands with uv
uv run python -m smolval.cli --help

# Or install globally
uv pip install -e .
smolval --help
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

### Example
```bash
# Show version
smolval --version

# Run with verbose logging
smolval --verbose eval prompts/test.txt

# Run with debug logging and no banner
smolval --debug --no-banner eval prompts/test.txt
```

## Commands

### eval

Evaluate MCP servers using a single prompt file.

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
| `--config, -c` | `config/example-anthropic.yaml` | Configuration file path |
| `--output-dir` | `results` | Output directory for results |
| `--run-name` | _(prompt filename)_ | Name for this evaluation run |

#### Examples

```bash
# Basic evaluation
smolval eval prompts/file-operations.txt

# With custom config and output directory
smolval eval prompts/web-test.txt -c config/my-config.yaml --output-dir custom-results

# With custom run name
smolval eval prompts/database.txt --run-name "db-performance-test"

# Verbose output with debug config
smolval --verbose eval prompts/test.txt -c config/debug.yaml
```

#### Output

Creates a timestamped directory in the output folder containing:
- `evaluation_{run_name}.json` - Structured evaluation data
- `evaluation_{run_name}.csv` - Tabular results
- `evaluation_{run_name}.md` - Markdown report
- `evaluation_{run_name}.html` - HTML report

### batch

Run multiple prompts from a directory in batch mode.

```
smolval batch PROMPTS_DIR [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `PROMPTS_DIR` | Path | Yes | Directory containing prompt files (.txt, .md) |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config, -c` | `config/example-anthropic.yaml` | Configuration file path |
| `--output-dir` | `results` | Output directory for results |
| `--run-name` | `batch_{directory_name}` | Name for this batch run |
| `--servers` | _(all configured)_ | Comma-separated server names to filter |

#### Examples

```bash
# Run all prompts in directory
smolval batch prompts/

# Filter to specific servers
smolval batch prompts/ --servers filesystem,fetch

# Custom batch name and output
smolval batch test-prompts/ --run-name "integration-tests" --output-dir ci-results

# Production batch with specific config
smolval batch prompts/ -c config/production.yaml --output-dir prod-results
```

#### Output

Creates individual evaluation files for each prompt plus:
- `batch_summary_{run_name}.json` - Batch execution summary
- `batch_summary_{run_name}.csv` - Success/failure statistics
- `batch_summary_{run_name}.md` - Batch report
- `batch_summary_{run_name}.html` - HTML batch report

#### Batch Behavior

- Processes all `.txt` and `.md` files in the specified directory
- Continues processing even if individual prompts fail
- Reports overall success rate and timing statistics
- Skips invalid or inaccessible files with warnings

### compare

Compare the performance of two MCP servers using the same set of prompts.

```
smolval compare --baseline BASELINE --test TEST PROMPTS_DIR [OPTIONS]
```

#### Required Options

| Option | Description |
|--------|-------------|
| `--baseline` | Name of the baseline server (must exist in config) |
| `--test` | Name of the test server to compare against baseline |

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `PROMPTS_DIR` | Path | Yes | Directory containing prompt files for comparison |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config, -c` | `config/example-anthropic.yaml` | Configuration file path |
| `--output-dir` | `results` | Output directory for comparison results |
| `--run-name` | `compare_{baseline}_vs_{test}` | Name for this comparison run |

#### Examples

```bash
# Basic server comparison
smolval compare --baseline filesystem --test fetch prompts/

# Compare database servers
smolval compare --baseline postgres --test postgresv2 prompts/db-tests/

# Custom comparison with specific config
smolval compare --baseline old-server --test new-server prompts/ \
  -c config/comparison.yaml --run-name "migration-test"

# Production comparison
smolval compare --baseline prod-v1 --test prod-v2 prompts/benchmarks/ \
  --output-dir comparison-results
```

#### Output

Creates comparison analysis files:
- `comparison_{run_name}.json` - Detailed comparison data
- `comparison_{run_name}.csv` - Comparison metrics table
- `comparison_{run_name}.md` - Comparison report
- `comparison_{run_name}.html` - Interactive comparison report

#### Comparison Metrics

The comparison analyzes:
- **Success Rate**: Percentage of successful evaluations
- **Average Execution Time**: Mean time per evaluation
- **Average Iterations**: Mean ReAct loop iterations
- **Failed Tool Calls**: Number of tool execution failures
- **Token Usage**: LLM token consumption (if available)
- **Overall Winner**: Best performing server across metrics

## Output Formats

All commands generate results in multiple formats:

### JSON Format
```json
{
  "evaluation_name": "test",
  "timestamp": "2024-01-15T10:30:00Z",
  "success": true,
  "execution_time": 12.5,
  "iterations": 3,
  "messages": [...],
  "tool_calls": [...],
  "performance_metrics": {...}
}
```

### CSV Format
Tabular data suitable for spreadsheet analysis with columns:
- evaluation_name, timestamp, success, execution_time, iterations, etc.

### Markdown Format
Human-readable reports with:
- Executive summary
- Tool call breakdown
- Performance metrics
- Full conversation log

### HTML Format
Interactive web reports with:
- Collapsible sections
- Syntax-highlighted code blocks
- Performance charts
- Responsive design

## Configuration

All commands accept a configuration file via the `--config` option. See [Configuration Reference](config-reference.md) for details.

### Default Configuration Lookup

If no config is specified, smolval looks for:
1. `config/example-anthropic.yaml` (default)
2. `config/default.yaml`
3. `smolval.yaml` in current directory

### Environment Variables

Configuration files support environment variable expansion:
```yaml
llm:
  api_key: "${ANTHROPIC_API_KEY}"
  base_url: "${API_BASE_URL:-https://api.anthropic.com}"
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (configuration, file not found, etc.) |
| 2 | Evaluation failure (timeout, server error, etc.) |
| 3 | No servers configured or available |

## Common Usage Patterns

### Development Workflow
```bash
# Quick test with debug output
smolval --debug eval prompts/dev-test.txt -c config/dev.yaml

# Batch test all prompts
smolval batch prompts/ --output-dir dev-results

# Compare development vs production servers
smolval compare --baseline dev --test prod prompts/
```

### CI/CD Integration
```bash
# Production evaluation with specific output location
smolval batch prompts/ -c config/ci.yaml --output-dir /var/log/smolval

# Server migration validation
smolval compare --baseline old --test new prompts/migration-tests/ \
  --output-dir migration-results --run-name "$(date +%Y%m%d)"
```

### Performance Testing
```bash
# Benchmark single server
smolval eval prompts/performance.txt --run-name "benchmark-$(date +%s)"

# Compare server performance
smolval compare --baseline v1 --test v2 prompts/benchmarks/ \
  --run-name "perf-comparison"
```

### Local Development with Ollama
```bash
# Test with local LLM (no API key required)
smolval eval prompts/test.txt -c config/ollama.yaml

# Batch test with local model
smolval batch prompts/ -c config/ollama.yaml --output-dir local-results
```

## Troubleshooting

### Common Issues

1. **Configuration not found**
   ```
   Error: Configuration file not found: config/missing.yaml
   ```
   Solution: Check file path or create the configuration file

2. **No MCP servers configured**
   ```
   Error: No MCP servers found in configuration
   ```
   Solution: Add MCP server definitions to your config file

3. **Server connection failed**
   ```
   Error: Failed to connect to MCP server 'filesystem'
   ```
   Solution: Verify server command and dependencies are installed

4. **API key missing**
   ```
   Error: No API key found for provider 'anthropic'
   ```
   Solution: Set environment variable or update config file

### Debug Mode

Use `--debug` flag for detailed troubleshooting:
```bash
smolval --debug eval prompts/test.txt
```

This enables:
- Detailed connection logs
- Full MCP protocol messages
- LLM request/response details
- Timing information

### Verbose Mode

Use `--verbose` for general progress information:
```bash
smolval --verbose batch prompts/
```

This shows:
- Server connection status
- Prompt processing progress
- Basic timing information
- Success/failure summaries

## Integration Examples

### Shell Scripts
```bash
#!/bin/bash
# run-evaluation.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/eval_${TIMESTAMP}"

smolval batch prompts/ \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "automated-eval" \
  -c config/production.yaml

echo "Results available in: ${OUTPUT_DIR}"
```

### Make Targets
```makefile
# Makefile
.PHONY: test-mcp test-compare

test-mcp:
	smolval batch prompts/ -c config/test.yaml --output-dir test-results

test-compare:
	smolval compare --baseline filesystem --test fetch prompts/ \
		--output-dir comparison-results
```

### Docker Integration
```bash
# Run in container
docker run --rm \
  -v $(pwd)/prompts:/prompts \
  -v $(pwd)/results:/results \
  -e ANTHROPIC_API_KEY \
  smolval batch /prompts --output-dir /results
```

This CLI reference covers all available commands and options. For configuration details, see [Configuration Reference](config-reference.md). For prompt writing guidance, see [Writing Prompts](writing-prompts.md).