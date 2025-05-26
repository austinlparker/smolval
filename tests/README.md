# Testing Guide for smolval

This guide explains how to run different types of tests in the smolval project.

## Test Types

### Unit Tests
Fast tests that test individual components in isolation using mocks.

```bash
# Run all unit tests (excludes integration tests)
uv run pytest -m "not integration" -v

# Run specific unit test files
uv run pytest tests/test_agent.py -v
uv run pytest tests/test_config.py -v
```

### Integration Tests

#### 1. Simple Integration Tests (Mock Everything)
Tests component integration using mocks for all external dependencies.

```bash
# Fast integration tests with all mocks
uv run pytest tests/test_integration_simple.py -v
```

#### 2. Real MCP Server Integration Tests
Tests using real MCP servers (NPM and Python packages) with mocked LLMs.

```bash
# Run real MCP server integration tests
uv run pytest tests/test_integration.py -v

# Run filesystem server tests (requires NPM)
uv run pytest tests/test_integration.py::TestFileSystemIntegration -v

# Run fetch server tests (requires uvx)
uv run pytest tests/test_integration.py::TestFetchIntegration -v
```

#### 3. Real LLM Integration Tests (Costs Money! 💰)
Tests using real LLM APIs - **requires API keys and will cost money**.

```bash
# ⚠️ WARNING: These tests make real API calls and cost money!

# Set up API keys first:
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Run real LLM tests (costs money!)
uv run pytest tests/test_integration_real_llm.py -v

# Run only Anthropic tests
uv run pytest tests/test_integration_real_llm.py::TestRealLLMIntegration::test_anthropic_with_mock_filesystem -v

# Skip expensive tests
uv run pytest -m "integration and not requires_api_keys" -v
```

### All Tests
```bash
# Run everything
uv run pytest -v

# Run with coverage
uv run pytest --cov=smolval --cov-report=html

# Run tests in parallel (faster)
uv run pytest -n auto
```

## Test Markers

Tests are marked with pytest markers to categorize them:

- `@pytest.mark.integration` - Integration tests (may be slower)
- `@pytest.mark.unit` - Unit tests (fast)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_docker` - Tests that need Docker
- `@pytest.mark.requires_npm` - Tests that need NPM packages  
- `@pytest.mark.requires_api_keys` - Tests that make real API calls (costs money)

### Running Tests by Marker

```bash
# Run only unit tests
uv run pytest -m unit

# Run integration tests excluding slow ones
uv run pytest -m "integration and not slow"

# Run tests that don't require external dependencies
uv run pytest -m "not requires_docker and not requires_npm and not requires_api_keys"
```

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures
│
# Unit Tests (Fast, All Mocked)
├── test_agent.py                # Agent unit tests ✅
├── test_config.py               # Configuration unit tests ✅ 
├── test_llm_client.py           # LLM client unit tests ✅
├── test_mcp_client.py           # MCP client unit tests ✅
│
# Integration Tests  
├── test_integration_simple.py   # Simple integration (all mocked) ✅
├── test_integration.py          # Real MCP server integration ✅
├── test_integration_real_llm.py # Real LLM integration (costs money) 💰
├── test_integration_ollama.py   # Ollama local model integration ✅
```

### Test Categories Explained

| Test File | LLM | MCP Servers | Speed | Cost | Purpose |
|-----------|-----|-------------|-------|------|---------|
| `test_*_unit.py` | Mocked | Mocked | ⚡ Fast | Free | Unit testing individual components |
| `test_integration_simple.py` | Mocked | Mocked | ⚡ Fast | Free | Component integration with mocks |
| `test_integration.py` | Mocked | **Real Servers** | 🐌 Medium | Free | Realistic server interaction (NPM/Python) |
| `test_integration_ollama.py` | **Ollama** | **Real Servers** | 🐌 Medium | Free | Local model testing |
| `test_integration_real_llm.py` | **Real APIs** | **Real Servers** | 🐌 Slow | 💰 **Costs Money** | End-to-end with real LLMs |

## Current Test Status

### ✅ All Tests Passing
- **Unit tests**: All component tests passing
- **Integration tests**: Real MCP server tests passing
- **LLM client tests**: Multi-provider support working
- **Configuration tests**: YAML loading and validation working
- **CLI tests**: End-to-end command line interface working

## Test Architecture Highlights

### Real MCP Server Integration
- **NPM Servers**: Tests use `@modelcontextprotocol/server-filesystem` 
- **Python Servers**: Tests use `mcp-server-fetch` via uvx
- **Docker Servers**: Support for `mcp/sqlite` (when Docker available)
- **Conditional Skipping**: Tests skip gracefully when dependencies unavailable

### Multi-Provider LLM Support
- **Anthropic Claude**: API integration with real and mock responses
- **OpenAI GPT**: API integration with proper error handling  
- **Ollama**: Local model integration with Gemma-specific function calling

## Running Specific Test Scenarios

### Development Workflow
```bash
# Quick unit test run during development
uv run pytest -m "not integration" --tb=short

# Test specific component you're working on
uv run pytest tests/test_agent.py::TestAgent::test_simple_task_no_tools -v

# Run integration tests to verify end-to-end functionality
uv run pytest tests/test_integration_simple.py -v
```

### CI/CD Pipeline
```bash
# Fast test suite for pull requests
uv run pytest -m "not slow" --maxfail=5

# Full test suite for main branch
uv run pytest --cov=smolval --cov-fail-under=80

# Integration tests only
uv run pytest -m integration --tb=short
```

### Debug Failing Tests
```bash
# Run with detailed output
uv run pytest tests/test_llm_client.py -vvv --tb=long

# Run single failing test with debug output
uv run pytest tests/test_llm_client.py::TestLLMClient::test_create_anthropic_client -s -vvv

# Run with PDB debugger on failure
uv run pytest tests/test_agent.py --pdb
```

## Real MCP Servers for Integration Testing

The project tests against real MCP servers to ensure compatibility:

### Filesystem Server (NPM)
```yaml
- name: "filesystem"
  command: ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"]
  env: {}
```

### Fetch Server (Python)
```yaml
- name: "fetch"
  command: ["uvx", "mcp-server-fetch"]
  env: {}
```

### SQLite Server (Docker)
```yaml
- name: "sqlite"
  command: ["docker", "run", "--rm", "-i", "-v", "mcp-test:/mcp", "mcp/sqlite", "--db-path", "/mcp/test.db"]
  env: {}
```

## Adding New Tests

### Unit Tests
1. Add test to appropriate `test_*.py` file
2. Use mocks for external dependencies
3. Mark with `@pytest.mark.unit` if needed
4. Keep tests fast (< 1 second)

### Integration Tests  
1. Add to `test_integration_simple.py` for basic integration
2. Add to `test_integration.py` for complex scenarios
3. Mark with `@pytest.mark.integration`
4. Use mock servers when possible
5. Mark as `@pytest.mark.slow` if > 5 seconds

### Example Unit Test
```python
@pytest.mark.unit
def test_new_feature(self):
    """Test new feature functionality."""
    # Setup
    config = create_test_config()
    
    # Execute
    result = new_feature(config)
    
    # Verify
    assert result.success
    assert result.data == expected_data
```

### Example Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_new_integration(self):
    """Test integration between components."""
    # Setup real components
    mcp_manager = MCPClientManager()
    
    try:
        # Test integration
        await mcp_manager.connect(test_config)
        result = await test_operation()
        
        # Verify
        assert result.success
    finally:
        await mcp_manager.close()
```

## Performance Testing

```bash
# Time test execution
uv run pytest --durations=10

# Profile memory usage
uv run pytest --profile

# Run tests with timeout
uv run pytest --timeout=300
```

## Troubleshooting

### Common Issues

1. **Tests hang**: Usually async test issues, check for missing `await` or `@pytest.mark.asyncio`
2. **Import errors**: Check `PYTHONPATH` and virtual environment
3. **Mock failures**: Verify mock interfaces match actual code
4. **Resource leaks**: Ensure proper cleanup in `finally` blocks

### Debug Environment
```bash
# Check test discovery
uv run pytest --collect-only

# Verify fixtures
uv run pytest --fixtures

# Check test configuration
uv run pytest --markers
```