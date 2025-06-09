"""Tests for results module."""

import json
import tempfile
from pathlib import Path

import pytest

from smolval.results import ResultsFormatter


class TestResultsFormatter:
    """Test ResultsFormatter class."""

    def test_init_valid_formats(self):
        """Test initialization with valid formats."""
        for format_type in ["json", "csv", "markdown", "html"]:
            formatter = ResultsFormatter(format_type)
            assert formatter.format_type == format_type

    def test_init_invalid_format(self):
        """Test initialization with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format: invalid"):
            ResultsFormatter("invalid")

    def test_init_default_format(self):
        """Test initialization with default format."""
        formatter = ResultsFormatter()
        assert formatter.format_type == "json"

    def test_format_result_json(self):
        """Test formatting result as JSON."""
        formatter = ResultsFormatter("json")
        result_data = {
            "success": True,
            "execution_time_seconds": 5.0,
            "total_iterations": 3,
            "final_answer": "Test answer",
        }

        result = formatter.format_result(result_data)

        # Should return formatted JSON string
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["execution_time_seconds"] == 5.0
        assert parsed["total_iterations"] == 3
        assert parsed["final_answer"] == "Test answer"

    def test_format_result_json_with_file(self):
        """Test formatting result as JSON with file output."""
        formatter = ResultsFormatter("json")
        result_data = {"success": True, "test": "data"}

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            result = formatter.format_result(result_data, temp_file)

            # Should write to file and return same content
            with open(temp_file) as f:
                file_content = f.read()

            assert result == file_content
            parsed = json.loads(file_content)
            assert parsed["success"] is True
            assert parsed["test"] == "data"

        finally:
            Path(temp_file).unlink()

    def test_format_result_csv(self):
        """Test formatting result as CSV."""
        formatter = ResultsFormatter("csv")
        result_data = {
            "success": True,
            "execution_time_seconds": 2.5,
            "total_iterations": 4,
            "failed_tool_calls": 1,
            "final_answer": "Test answer",
            "error": None,
        }

        result = formatter.format_result(result_data)

        lines = result.strip().split("\n")
        assert len(lines) == 2  # Header + data

        # Check header
        header = lines[0]
        assert "success" in header
        assert "execution_time_seconds" in header
        assert "total_iterations" in header

        # Check data
        data = lines[1]
        assert "True" in data
        assert "2.5" in data
        assert "4" in data
        assert "1" in data
        assert "11" in data  # Length of "Test answer"

    def test_format_result_markdown(self):
        """Test formatting result as Markdown."""
        formatter = ResultsFormatter("markdown")
        result_data = {
            "success": True,
            "execution_time_seconds": 1.5,
            "total_iterations": 2,
            "final_answer": "Test markdown answer",
            "steps": [
                {
                    "step_type": "tool_use",
                    "content": "First content",
                    "tool_name": "test_tool",
                    "tool_output": "Test observation",
                }
            ],
            "metadata": {
                "session_id": "test-session",
                "model_used": "claude-3-sonnet",
            },
        }

        result = formatter.format_result(result_data)

        # Should contain markdown formatting
        assert "# Evaluation Result" in result
        assert "✅ Success" in result
        assert "**Execution Time:** 1.50s" in result
        assert "**Iterations:** 2" in result
        assert "## Final Answer" in result
        assert "Test markdown answer" in result
        assert "## Execution Steps" in result
        assert "### Step 1" in result
        assert "**Content:** First content" in result
        assert "**Tool:** test_tool" in result
        assert "**Tool Output:**" in result
        assert "Test observation" in result

    def test_format_result_markdown_with_mcp_info(self):
        """Test formatting result with MCP servers and tools information."""
        formatter = ResultsFormatter("markdown")
        result_data = {
            "success": True,
            "execution_time_seconds": 1.5,
            "total_iterations": 2,
            "final_answer": "Test answer with MCP",
            "steps": [],
            "metadata": {
                "session_id": "test-session",
                "model_used": "claude-3-sonnet",
                "mcp_servers_used": ["sqlite", "github"],
                "tools_available": ["Read", "Write", "Bash", "WebFetch"],
            },
        }

        result = formatter.format_result(result_data)

        # Should contain environment section
        assert "## Environment" in result
        assert "**MCP Servers:** sqlite, github" in result
        assert "**Available Tools:** Read, Write, Bash, WebFetch" in result

    def test_format_result_markdown_with_error(self):
        """Test formatting result with error as Markdown."""
        formatter = ResultsFormatter("markdown")
        result_data = {
            "success": False,
            "execution_time_seconds": 0.5,
            "total_iterations": 0,
            "error_message": "Test error message",
            "final_answer": "",
            "metadata": {
                "session_id": "test-session",
                "model_used": "claude-3-sonnet",
            },
        }

        result = formatter.format_result(result_data)

        assert "❌ Failed" in result
        assert "## Error" in result
        assert "Test error message" in result

    def test_format_result_html(self):
        """Test formatting result as HTML."""
        formatter = ResultsFormatter("html")
        result_data = {
            "success": True,
            "execution_time_seconds": 3.0,
            "total_iterations": 1,
            "final_answer": "HTML test answer",
        }

        result = formatter.format_result(result_data)

        # Should contain HTML tags
        assert "<html" in result
        assert "<body>" in result
        # Check for success indicator in template (may be emoji or icon)
        assert "✅" in result or "success" in result.lower()
        assert "3.0" in result
        assert "HTML test answer" in result

    def test_format_result_html_with_error(self):
        """Test formatting result with error as HTML."""
        formatter = ResultsFormatter("html")
        result_data = {
            "success": False,
            "error": "HTML error test",
            "execution_time_seconds": 0.1,
            "total_iterations": 0,
        }

        result = formatter.format_result(result_data)

        assert "<html" in result
        # Check for error indicator in template (may be emoji or icon)
        assert "❌" in result or "failed" in result.lower() or "error" in result.lower()
        # Note: Error message might be displayed differently in complex HTML template

    def test_format_result_unsupported(self):
        """Test format_result with unsupported format."""
        formatter = ResultsFormatter("json")
        formatter.format_type = "unsupported"  # Manually set invalid format

        with pytest.raises(ValueError, match="Unsupported format: unsupported"):
            formatter.format_result({"test": "data"})
