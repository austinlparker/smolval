"""Tests for config module."""

from smolval.config import Config


class TestConfig:
    """Test configuration functionality."""

    def test_config_default(self):
        """Test default configuration."""
        config = Config.default()
        assert config.timeout_seconds == 300
        assert config.output_format == "markdown"

    def test_config_custom(self):
        """Test custom configuration."""
        config = Config(timeout_seconds=120, output_format="json")
        assert config.timeout_seconds == 120
        assert config.output_format == "json"

    def test_config_validation_timeout(self):
        """Test timeout validation."""
        # Negative timeout should be allowed by the model, but might not make sense
        config = Config(timeout_seconds=-1)
        assert config.timeout_seconds == -1

    def test_config_validation_format(self):
        """Test output format validation."""
        # Valid formats should work
        for fmt in ["json", "csv", "markdown", "html"]:
            config = Config(output_format=fmt)
            assert config.output_format == fmt
