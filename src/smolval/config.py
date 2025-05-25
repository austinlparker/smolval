"""Configuration system for smolval."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    
    name: str = Field(..., description="Unique name for the MCP server")
    command: List[str] = Field(..., description="Command to start the server")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    @field_validator('command')
    @classmethod
    def command_not_empty(cls, v: List[str]) -> List[str]:
        """Validate command is not empty."""
        if not v:
            raise ValueError("Command cannot be empty")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(..., description="LLM provider (anthropic, openai)")
    model: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key for the provider")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    
    @field_validator('provider')
    @classmethod
    def valid_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        if v not in ("anthropic", "openai"):
            raise ValueError("Provider must be 'anthropic' or 'openai'")
        return v
    
    @field_validator('temperature')
    @classmethod
    def valid_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for evaluation parameters."""
    
    timeout_seconds: int = Field(default=60, description="Timeout for evaluations")
    max_iterations: int = Field(default=10, description="Maximum agent loop iterations")
    output_format: str = Field(default="json", description="Output format (json, csv, markdown)")
    
    @field_validator('timeout_seconds')
    @classmethod
    def positive_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @field_validator('max_iterations')
    @classmethod
    def positive_iterations(cls, v: int) -> int:
        """Validate max_iterations is positive."""
        if v <= 0:
            raise ValueError("Max iterations must be positive")
        return v
    
    @field_validator('output_format')
    @classmethod
    def valid_format(cls, v: str) -> str:
        """Validate output format is supported."""
        if v not in ("json", "csv", "markdown"):
            raise ValueError("Output format must be 'json', 'csv', or 'markdown'")
        return v


class Config(BaseModel):
    """Main configuration for smolval."""
    
    mcp_servers: List[MCPServerConfig] = Field(..., description="MCP servers to connect to")
    llm: LLMConfig = Field(..., description="LLM configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation settings")
    
    @field_validator('mcp_servers')
    @classmethod
    def at_least_one_server(cls, v: List[MCPServerConfig]) -> List[MCPServerConfig]:
        """Validate at least one MCP server is configured."""
        if not v:
            raise ValueError("At least one MCP server must be configured")
        return v
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary with environment variable expansion."""
        # Expand environment variables
        expanded_data = cls._expand_env_vars(data)
        return cls(**expanded_data)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @staticmethod
    def _expand_env_vars(obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: Config._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Expand ${VAR} and ${VAR:-default} patterns
            def replace_var(match: re.Match[str]) -> str:
                var_expr = match.group(1)
                if ':-' in var_expr:
                    var_name, default = var_expr.split(':-', 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(var_expr, match.group(0))
            
            return re.sub(r'\$\{([^}]+)\}', replace_var, obj)
        else:
            return obj