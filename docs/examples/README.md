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

- **[development-config.yaml](development-config.yaml)** - Simple setup for development and testing (legacy format)
- **Basic MCP Setup** - Claude Code includes filesystem and web fetch capabilities built-in

### Database Configurations  

- **[postgres/postgres-comparison.yaml](postgres/postgres-comparison.yaml)** - PostgreSQL database access via MCP
- **Standard `.mcp.json`** - Uses standard MCP configuration format

### Production Configurations

- **[production-config.yaml](production-config.yaml)** - Production-ready setup (legacy format)
- **Docker MCP Servers** - Containerized MCP servers for production isolation

## Quick Start

### 1. Basic File Operations Test (No MCP Config Needed)

```bash
# Test basic filesystem operations (built into Claude Code)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/basic-file-operations.txt
```

### 2. Web Content Analysis (No MCP Config Needed)

```bash
# Test web fetching capabilities (built into Claude Code)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/web-content-analysis.txt --format html
```

### 3. Database Operations (Requires MCP Config)

```bash
# Create .mcp.json for database access
cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    }
  }
}
EOF

# Test database operations with MCP server
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/database-operations.txt --mcp-config /workspace/.mcp.json
```

### 4. Performance Benchmark

```bash
# Run comprehensive performance tests
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/performance-benchmark.txt --format html --timeout 600
```

## Customizing Examples

### Modifying Prompts

All prompt files can be customized for your specific use case:

1. **Copy the example**: `cp docs/examples/basic-file-operations.txt prompts/my-test.txt`
2. **Edit the tasks**: Modify the numbered steps to match your requirements
3. **Update success criteria**: Adjust the performance and correctness expectations
4. **Run your test**: 
   ```bash
   docker run --rm \
     -v $(pwd):/workspace \
     -e ANTHROPIC_API_KEY \
     ghcr.io/austinlparker/smolval \
     eval /workspace/prompts/my-test.txt
   ```

### Adapting MCP Configurations

Example `.mcp.json` configurations for different environments:

```json
// Development - simple setup
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

```json
// Production - multiple servers
{
  "mcpServers": {
    "database": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@host:5432/db"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

## Environment-Specific Examples

### Local Development

```bash
# Set up environment
export ANTHROPIC_API_KEY="your-key-here"

# Run basic tests (no MCP servers needed)
docker run --rm \
  -v $(pwd):/workspace \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/basic-file-operations.txt
```

### CI/CD Pipeline

```bash
# Use in continuous integration
docker run --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e ANTHROPIC_API_KEY \
  ghcr.io/austinlparker/smolval \
  eval /workspace/docs/examples/performance-benchmark.txt \
  --format json --output ci-results-${BUILD_NUMBER}.json
```

### Batch Testing Multiple Prompts

```bash
# Test multiple prompts in sequence
for prompt in docs/examples/*.txt; do
  echo "Testing $prompt..."
  docker run --rm \
    -v $(pwd):/workspace \
    -e ANTHROPIC_API_KEY \
    ghcr.io/austinlparker/smolval \
    eval "/workspace/$prompt" --format json
done
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

### MCP Configuration Template

```json
{
  "mcpServers": {
    "descriptive-name": {
      "command": "executable",
      "args": ["arg1", "arg2"],
      "env": {
        "KEY": "value"
      }
    },
    "database": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    },
    "api-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

**Note**: Claude Code handles the LLM configuration automatically using your `ANTHROPIC_API_KEY`. Timeout and output format are controlled via command-line flags.

## Troubleshooting Examples

If examples don't work as expected:

1. **Check API key**: Ensure `ANTHROPIC_API_KEY` is set correctly
2. **Verify file paths**: Ensure prompt files exist in `/workspace/` within container
3. **Run with verbose mode**: Use `--verbose` flag for detailed logs
4. **Check Docker setup**: Ensure Docker daemon is running for MCP servers
5. **Test without MCP**: Try basic examples first (filesystem and web fetch work without MCP)
6. **Check MCP config**: Validate `.mcp.json` syntax with `python3 -m json.tool .mcp.json`

## Contributing Examples

To contribute new examples:

1. Create meaningful prompt files that test specific functionality
2. Provide configuration files that demonstrate best practices
3. Include clear documentation and expected outcomes
4. Test examples across different environments
5. Submit a pull request with your additions

These examples provide a solid foundation for evaluating MCP servers and can be adapted for various testing scenarios.