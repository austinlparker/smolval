"""Tests for configuration system."""

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from smolval.config import Config, MCPServerConfig, LLMConfig, EvaluationConfig


class TestConfig:
    """Test configuration loading and validation."""

    def test_config_from_dict(self, sample_config: Dict[str, Any]) -> None:
        """Test creating config from dictionary."""
        config = Config.from_dict(sample_config)
        
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].name == "filesystem"
        assert config.llm.provider == "anthropic"
        assert config.evaluation.timeout_seconds == 60

    def test_config_from_yaml_file(self, sample_config: Dict[str, Any]) -> None:
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = Path(f.name)
        
        try:
            config = Config.from_yaml(config_path)
            assert config.llm.model == "anthropic/claude-3-haiku-20240307"
        finally:
            config_path.unlink()

    def test_config_validation_missing_mcp_servers(self) -> None:
        """Test validation fails when mcp_servers is missing."""
        invalid_config = {
            "llm": {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "api_key": "test"},
            "evaluation": {"timeout_seconds": 60}
        }
        
        with pytest.raises(ValueError, match="mcp_servers"):
            Config.from_dict(invalid_config)

    def test_config_validation_invalid_llm_provider(self) -> None:
        """Test validation fails with invalid LLM provider."""
        invalid_config = {
            "mcp_servers": [{"name": "test", "command": ["echo"], "env": {}}],
            "llm": {"provider": "invalid", "model": "test", "api_key": "test"},
            "evaluation": {"timeout_seconds": 60}
        }
        
        with pytest.raises(ValueError, match="provider"):
            Config.from_dict(invalid_config)

    def test_mcp_server_config_validation(self) -> None:
        """Test MCP server configuration validation."""
        # Valid config
        server_config = MCPServerConfig(
            name="test",
            command=["python", "-m", "test_server"],
            env={"VAR": "value"}
        )
        assert server_config.name == "test"
        assert len(server_config.command) == 3
        
        # Invalid empty command
        with pytest.raises(ValueError):
            MCPServerConfig(name="test", command=[], env={})

    def test_llm_config_defaults(self) -> None:
        """Test LLM configuration with defaults."""
        llm_config = LLMConfig(provider="anthropic", model="claude-3-sonnet-20240229", api_key="test")
        
        assert llm_config.temperature == 0.1  # default
        assert llm_config.max_tokens == 1000  # default

    def test_evaluation_config_defaults(self) -> None:
        """Test evaluation configuration with defaults."""
        eval_config = EvaluationConfig()
        
        assert eval_config.timeout_seconds == 60
        assert eval_config.max_iterations == 10
        assert eval_config.output_format == "json"

    def test_environment_variable_expansion(self) -> None:
        """Test environment variable expansion in config."""
        import os
        
        # Set test environment variable
        test_key = "TEST_SMOLVAL_API_KEY"
        test_value = "secret123"
        os.environ[test_key] = test_value
        
        try:
            config_dict = {
                "mcp_servers": [{
                    "name": "test",
                    "command": ["echo"],
                    "env": {"API_KEY": f"${{{test_key}}}"}
                }],
                "llm": {"provider": "anthropic", "model": "test", "api_key": "test"},
                "evaluation": {}
            }
            
            config = Config.from_dict(config_dict)
            assert config.mcp_servers[0].env["API_KEY"] == test_value
        finally:
            del os.environ[test_key]

    def test_config_file_not_found(self) -> None:
        """Test error handling when config file doesn't exist."""
        non_existent_path = Path("/does/not/exist.yaml")
        
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(non_existent_path)