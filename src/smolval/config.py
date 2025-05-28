"""Configuration system for smolval."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args: object, **kwargs: object) -> bool:  # type: ignore[misc]
        """Stub function when python-dotenv is not available."""
        return False


# Load environment variables from a .env file if present
load_dotenv()


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str = Field(..., description="Unique name for the MCP server")
    command: list[str] = Field(..., description="Command to start the server")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: list[str]) -> list[str]:
        """Validate command is not empty."""
        if not v:
            raise ValueError("Command cannot be empty")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: str = Field(
        ..., description="LLM provider (anthropic, openai, ollama, gemini)"
    )
    model: str = Field(..., description="Model name")
    api_key: str | None = Field(
        default=None, description="API key for the provider (not required for ollama)"
    )
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    base_url: str | None = Field(
        default=None, description="Base URL for local providers like Ollama"
    )

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        if v not in ("anthropic", "openai", "ollama", "gemini"):
            raise ValueError(
                "Provider must be 'anthropic', 'openai', 'ollama', or 'gemini'"
            )
        return v

    @model_validator(mode="after")
    def validate_api_key_for_cloud_providers(self) -> "LLMConfig":
        """Validate API key is provided for cloud providers."""
        if self.provider in ("anthropic", "openai", "gemini") and not self.api_key:
            raise ValueError(f"API key is required for {self.provider}")
        return self

    @field_validator("temperature")
    @classmethod
    def valid_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class JudgeConfig(BaseModel):
    """Configuration for LLM-as-judge evaluation."""

    enabled: bool = Field(default=False, description="Enable LLM-as-judge evaluation")
    model: str | None = Field(
        default=None, description="Model to use for judgment (uses main LLM if None)"
    )
    provider: str | None = Field(
        default=None, description="Provider to use for judgment (uses main LLM if None)"
    )
    criteria_weights: dict[str, float] | None = Field(
        default=None, description="Custom weights for judgment criteria"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation parameters."""

    timeout_seconds: int = Field(default=60, description="Timeout for evaluations")
    max_iterations: int = Field(default=10, description="Maximum agent loop iterations")
    output_format: str = Field(
        default="json", description="Output format (json, csv, markdown)"
    )
    judge: JudgeConfig = Field(
        default_factory=JudgeConfig, description="LLM-as-judge settings"
    )

    @field_validator("timeout_seconds")
    @classmethod
    def positive_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @field_validator("max_iterations")
    @classmethod
    def positive_iterations(cls, v: int) -> int:
        """Validate max_iterations is positive."""
        if v <= 0:
            raise ValueError("Max iterations must be positive")
        return v

    @field_validator("output_format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        """Validate output format is supported."""
        if v not in ("json", "csv", "markdown"):
            raise ValueError("Output format must be 'json', 'csv', or 'markdown'")
        return v


class Config(BaseModel):
    """Main configuration for smolval."""

    mcp_servers: list[MCPServerConfig] = Field(
        ..., description="MCP servers to connect to"
    )
    llm: LLMConfig = Field(..., description="LLM configuration")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation settings"
    )
    allow_empty_servers: bool = Field(
        default=False, description="Allow empty MCP server list (for testing)"
    )

    @model_validator(mode="after")
    def validate_mcp_servers(self) -> "Config":
        """Validate at least one MCP server is configured (unless allow_empty_servers is True)."""
        if not self.mcp_servers and not self.allow_empty_servers:
            raise ValueError("At least one MCP server must be configured")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary with environment variable expansion."""
        # Expand environment variables
        expanded_data = cls._expand_env_vars(data)
        return cls(**expanded_data)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
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
                if ":-" in var_expr:
                    var_name, default = var_expr.split(":-", 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(var_expr, match.group(0))

            return re.sub(r"\$\{([^}]+)\}", replace_var, obj)
        else:
            return obj
