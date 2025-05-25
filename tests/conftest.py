"""Pytest configuration and fixtures for smolval tests."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest


# Remove custom event_loop fixture to avoid deprecation warning
# pytest-asyncio will handle this automatically


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_prompt() -> str:
    """Return a sample evaluation prompt."""
    return """
    Please use the available MCP tools to:
    1. List files in the current directory
    2. Read the contents of any .txt files found
    3. Summarize what you discovered
    
    Expected outcome: Successfully demonstrate file system operations.
    """


@pytest.fixture
def sample_config() -> dict:
    """Return a sample configuration dictionary."""
    return {
        "mcp_servers": [
            {
                "name": "filesystem",
                "command": ["python", "-m", "mcp_server_filesystem", "/tmp"],
                "env": {}
            }
        ],
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "test-api-key",
            "temperature": 0.1
        },
        "evaluation": {
            "timeout_seconds": 60,
            "max_iterations": 10,
            "output_format": "json"
        }
    }