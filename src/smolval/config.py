"""Minimal configuration system for smolval."""

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Minimal configuration for smolval output settings."""

    timeout_seconds: int = Field(default=300, description="Timeout for evaluations")
    output_format: str = Field(
        default="markdown", description="Output format (json, csv, markdown, html)"
    )

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()
