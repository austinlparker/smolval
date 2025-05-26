# Configuration Reference

This document provides a comprehensive reference for all configuration options in smolval.

## Configuration File Structure

smolval uses YAML configuration files with the following top-level sections:

```yaml
mcp_servers:       # MCP server definitions
  - name: "server1"
    command: [...]
    env: {}

llm:              # LLM provider configuration
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"

evaluation:       # Evaluation parameters
  timeout_seconds: 120
  max_iterations: 15
  output_format: "json"
```

## Environment Variable Expansion

Configuration files support environment variable expansion:

- `${VAR}` - Required variable (fails if not set)
- `${VAR:-default}` - Optional variable with default value
- Variables are loaded from `.env` file if present

Example:
```yaml
llm:
  api_key: "${ANTHROPIC_API_KEY}"
  base_url: "${API_BASE_URL:-https://api.anthropic.com}"
```

## MCP Servers Configuration

### Basic Server Definition

```yaml
mcp_servers:
  - name: "server-name"           # Required: Unique identifier
    command: ["cmd", "arg1"]      # Required: Command to start server
    env: {}                       # Optional: Environment variables
```

### NPM-based Servers

```yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    env: {}
```

### Python-based Servers

```yaml
mcp_servers:
  - name: "fetch"
    command: ["uvx", "mcp-server-fetch"]
    env:
      USER_AGENT: "smolval-evaluator"
```

### Docker-based Servers

```yaml
mcp_servers:
  - name: "sqlite"
    command: ["docker", "run", "--rm", "-i", "-v", "data:/data", "mcp/sqlite", "--db-path", "/data/test.db"]
    env: {}
```

### Advanced Server Configuration

```yaml
mcp_servers:
  - name: "complex-server"
    command: ["python", "-m", "my_mcp_server"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
      LOG_LEVEL: "debug"
      CUSTOM_CONFIG: "/path/to/config.json"
```

## LLM Provider Configuration

### Anthropic Claude

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"    # or claude-3-haiku-20240307
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1                       # 0.0 to 1.0
  max_tokens: 4096                       # Optional
  base_url: "https://api.anthropic.com"  # Optional
```

Available models:
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`

### OpenAI

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"                   # or gpt-4, gpt-3.5-turbo
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.1
  max_tokens: 4096
  base_url: "https://api.openai.com/v1"  # Optional
```

Available models:
- `gpt-4o-mini` (recommended for cost)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Ollama (Local)

```yaml
llm:
  provider: "ollama"
  model: "gemma2:2b"                     # or llama3, codellama, etc.
  base_url: "http://localhost:11434"
  temperature: 0.1
  # No API key required for local Ollama
```

Common Ollama models:
- `gemma2:2b` (recommended for speed)
- `llama3:8b`
- `codellama:7b`
- `mistral:7b`

### Azure OpenAI

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${AZURE_OPENAI_API_KEY}"
  base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  api_version: "2023-12-01-preview"
```

## Evaluation Configuration

### Basic Settings

```yaml
evaluation:
  timeout_seconds: 120        # Maximum time for entire evaluation
  max_iterations: 15          # Maximum ReAct loop iterations
  output_format: "json"       # Default output format
  results_dir: "results"      # Directory for output files
```

### Advanced Settings

```yaml
evaluation:
  timeout_seconds: 300
  max_iterations: 20
  output_format: "markdown"
  results_dir: "custom-results"
  
  # Retry configuration
  max_retries: 3
  retry_delay: 1.0
  
  # Performance settings
  parallel_evaluations: 4
  stream_responses: true
  
  # Logging
  log_level: "INFO"           # DEBUG, INFO, WARNING, ERROR
  log_conversations: true     # Save full conversations
  
  # Output customization
  include_timestamps: true
  include_performance_metrics: true
  save_intermediate_results: true
```

## Output Format Options

### JSON Format
```yaml
evaluation:
  output_format: "json"
  json_options:
    indent: 2
    sort_keys: true
    ensure_ascii: false
```

### Markdown Format
```yaml
evaluation:
  output_format: "markdown"
  markdown_options:
    include_toc: true
    code_theme: "github"
    include_metadata: true
```

### HTML Format
```yaml
evaluation:
  output_format: "html"
  html_options:
    theme: "default"          # default, dark, minimal
    include_css: true
    collapsible_sections: true
```

### CSV Format
```yaml
evaluation:
  output_format: "csv"
  csv_options:
    delimiter: ","
    include_headers: true
    quote_style: "minimal"
```

## Complete Configuration Examples

### Development Configuration

```yaml
# config/development.yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "."]
    env: {}

llm:
  provider: "anthropic"
  model: "claude-3-haiku-20240307"  # Cheaper for development
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1

evaluation:
  timeout_seconds: 60
  max_iterations: 10
  output_format: "markdown"
  log_level: "DEBUG"
```

### Production Configuration

```yaml
# config/production.yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/data"]
    env: {}
  - name: "fetch"
    command: ["uvx", "mcp-server-fetch"]
    env:
      USER_AGENT: "smolval-prod"
  - name: "sqlite"
    command: ["docker", "run", "--rm", "-i", "-v", "db:/data", "mcp/sqlite", "--db-path", "/data/prod.db"]
    env: {}

llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1
  max_tokens: 8192

evaluation:
  timeout_seconds: 300
  max_iterations: 20
  output_format: "html"
  results_dir: "/var/log/smolval"
  log_level: "INFO"
  include_performance_metrics: true
```

### Multi-Provider Comparison

```yaml
# config/comparison.yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env: {}

# Primary LLM for evaluation
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1

# Alternative configurations for comparison
llm_alternatives:
  openai:
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.1
  
  ollama:
    provider: "ollama"
    model: "gemma2:2b"
    base_url: "http://localhost:11434"
    temperature: 0.1

evaluation:
  timeout_seconds: 180
  max_iterations: 15
  output_format: "html"
  compare_providers: true
```

### Minimal Configuration

```yaml
# config/minimal.yaml
mcp_servers:
  - name: "filesystem"
    command: ["npx", "@modelcontextprotocol/server-filesystem", "."]

llm:
  provider: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"

# All other settings use defaults
```

## Environment Variables

### Required Variables

- `ANTHROPIC_API_KEY` - For Anthropic Claude
- `OPENAI_API_KEY` - For OpenAI models
- `AZURE_OPENAI_API_KEY` - For Azure OpenAI

### Optional Variables

- `API_BASE_URL` - Custom API endpoint
- `LOG_LEVEL` - Override logging level
- `RESULTS_DIR` - Override results directory
- `USER_AGENT` - Custom user agent for web requests

### .env File Example

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...
LOG_LEVEL=INFO
RESULTS_DIR=./custom-results
```

## Validation and Error Handling

### Configuration Validation

smolval validates configuration files on startup:

- Required fields must be present
- API keys must be valid format
- Command paths must be accessible
- Numeric values must be in valid ranges

### Common Validation Errors

```yaml
# Error: Missing required field
mcp_servers:
  - command: ["npx", "server"]  # Missing 'name' field

# Error: Invalid timeout value
evaluation:
  timeout_seconds: -1          # Must be positive

# Error: Invalid model name
llm:
  provider: "anthropic"
  model: "invalid-model"       # Must be valid model ID
```

### Error Recovery

When configuration errors occur:

1. Detailed error messages are displayed
2. Invalid servers are skipped (with warnings)
3. Default values are used for optional settings
4. Evaluation continues if at least one server is valid

## Best Practices

### Security
- Store API keys in environment variables, not config files
- Use minimal permissions for file system access
- Regularly rotate API keys
- Use read-only database connections when possible

### Performance
- Use faster models (Haiku, GPT-3.5) for development
- Set appropriate timeouts for your use case
- Limit max_iterations for complex evaluations
- Use parallel evaluations for batch processing

### Maintainability
- Use descriptive server names
- Comment complex configurations
- Version control your config files
- Test configurations before production use

### Development
- Use separate configs for dev/staging/prod
- Start with minimal configs and add complexity
- Use debug logging during development
- Test with multiple LLM providers for compatibility