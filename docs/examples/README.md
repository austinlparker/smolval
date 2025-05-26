# smolval Examples

This directory contains example prompts and configurations to help you get started with smolval.

## Example Prompts

### Basic Functionality Tests

- **[basic-file-operations.txt](basic-file-operations.txt)** - Test filesystem server with basic file operations
- **[web-content-analysis.txt](web-content-analysis.txt)** - Test web fetch server with HTTP requests and content analysis
- **[database-operations.txt](database-operations.txt)** - Test database servers with various SQL operations

### Advanced Tests

- **[performance-benchmark.txt](performance-benchmark.txt)** - Comprehensive performance testing for MCP servers
- **[integration-test.txt](integration-test.txt)** - Multi-server coordination and real-world usage patterns

### Specialized Tests

- **[postgres/](postgres/)** - Complete PostgreSQL testing suite with test database, configurations, and comprehensive evaluation prompts

## Example Configurations

### Development Configurations

- **[config-basic.yaml](config-basic.yaml)** - Simple setup for development and testing
  - Filesystem server (current directory)
  - Web fetch server
  - Anthropic Claude with basic settings

### Comparison Configurations

- **[config-comparison.yaml](config-comparison.yaml)** - Setup for comparing different server implementations
  - Multiple filesystem servers with different configurations
  - Multiple fetch servers with optimization settings
  - Deterministic LLM settings for fair comparison

### Production Configurations

- **[config-production.yaml](config-production.yaml)** - Production-ready setup with security and performance optimizations
  - Restricted filesystem access
  - Production-grade web fetch limits
  - Database servers (PostgreSQL, SQLite)
  - Robust error handling and timeouts

## Quick Start

### 1. Basic File Operations Test

```bash
# Copy the basic config to your project
cp docs/examples/config-basic.yaml config/my-config.yaml

# Run the basic file operations test
uv run smolval eval docs/examples/basic-file-operations.txt -c config/my-config.yaml
```

### 2. Web Content Analysis

```bash
# Test web fetching capabilities
uv run smolval eval docs/examples/web-content-analysis.txt -c config/my-config.yaml
```

### 3. Performance Benchmark

```bash
# Run performance tests
uv run smolval eval docs/examples/performance-benchmark.txt -c config/my-config.yaml --format html
```

### 4. Server Comparison

```bash
# Compare two filesystem implementations
uv run smolval compare \
  --baseline filesystem-v1 \
  --test filesystem-v2 \
  docs/examples/ \
  -c docs/examples/config-comparison.yaml \
  --format html
```

## Customizing Examples

### Modifying Prompts

All prompt files can be customized for your specific use case:

1. **Copy the example**: `cp docs/examples/basic-file-operations.txt prompts/my-test.txt`
2. **Edit the tasks**: Modify the numbered steps to match your requirements
3. **Update success criteria**: Adjust the performance and correctness expectations
4. **Run your test**: `uv run smolval eval prompts/my-test.txt`

### Adapting Configurations

Example configurations can be adapted for different environments:

```yaml
# Development - fast and permissive
evaluation:
  timeout_seconds: 60
  max_iterations: 10
  retry_attempts: 1

# Production - robust and secure
evaluation:
  timeout_seconds: 300
  max_iterations: 25
  retry_attempts: 3
  retry_delay: 2.0
```

## Environment-Specific Examples

### Local Development

```bash
# Set up environment
export ANTHROPIC_API_KEY="your-key-here"

# Run basic tests
uv run smolval batch docs/examples/ -c docs/examples/config-basic.yaml
```

### CI/CD Pipeline

```bash
# Use in continuous integration
uv run smolval batch docs/examples/ \
  -c docs/examples/config-production.yaml \
  --output-dir ci-results/ \
  --run-name "ci-${BUILD_NUMBER}"
```

### Docker Environment

```bash
# Run in containerized environment
docker run --rm \
  -v $(pwd)/docs/examples:/examples \
  -v $(pwd)/results:/results \
  -e ANTHROPIC_API_KEY \
  smolval batch /examples/ \
  -c /examples/config-basic.yaml \
  --output-dir /results
```

## Creating Your Own Examples

### Prompt Template

```
[Brief description of what the test does]

[Step-by-step instructions:]
1. [Specific action to take]
2. [Another specific action]
3. [Verification step]

Success criteria:
- [Measurable success condition]
- [Performance expectation]
- [Error handling requirement]
```

### Configuration Template

```yaml
mcp_servers:
  - name: "descriptive-name"
    command: ["executable", "args"]
    env:
      KEY: "value"

llm:
  provider: "anthropic"  # or "openai", "ollama"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${API_KEY}"
  temperature: 0.1

evaluation:
  timeout_seconds: 120
  max_iterations: 15
  output_format: "markdown"
```

## Troubleshooting Examples

If examples don't work as expected:

1. **Check dependencies**: Ensure MCP servers are installed
2. **Verify API keys**: Check environment variables
3. **Run with debug**: Use `--debug` flag for detailed logs
4. **Check file paths**: Ensure prompt files and configs exist
5. **Review server logs**: Look for MCP server startup issues

## Contributing Examples

To contribute new examples:

1. Create meaningful prompt files that test specific functionality
2. Provide configuration files that demonstrate best practices
3. Include clear documentation and expected outcomes
4. Test examples across different environments
5. Submit a pull request with your additions

These examples provide a solid foundation for evaluating MCP servers and can be adapted for various testing scenarios.