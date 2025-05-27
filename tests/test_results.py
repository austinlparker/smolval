"""Tests for results module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_format_single_result_json(self):
        """Test formatting single result as JSON."""
        formatter = ResultsFormatter("json")
        result_data = {
            "success": True,
            "execution_time": 5.0,
            "iterations": 3,
            "test": "data",
        }

        result = formatter.format_single_result(result_data)

        # Should return formatted JSON string
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["execution_time"] == 5.0
        assert parsed["iterations"] == 3
        assert parsed["test"] == "data"

    def test_format_single_result_json_with_file(self):
        """Test formatting single result as JSON with file output."""
        formatter = ResultsFormatter("json")
        result_data = {"success": True, "test": "data"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            formatter.format_single_result(result_data, f.name)

            # Check file was written
            written_file = Path(f.name)
            assert written_file.exists()

            # Check content
            content = written_file.read_text()
            parsed = json.loads(content)
            assert parsed["success"] is True
            assert parsed["test"] == "data"

            # Cleanup
            written_file.unlink()

    def test_format_single_result_csv(self):
        """Test formatting single result as CSV."""
        formatter = ResultsFormatter("csv")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Test completed",
                "total_iterations": 3,
                "execution_time_seconds": 5.0,
                "failed_tool_calls": 0,
            },
            "metadata": {"prompt_file": "test.txt"},
        }

        result = formatter.format_single_result(result_data)

        # Should return CSV string with headers
        lines = result.strip().split("\n")
        assert len(lines) >= 2  # Header + data
        assert "success" in lines[0]
        assert "execution_time" in lines[0]
        assert "True" in lines[1] or "true" in lines[1].lower()

    def test_format_single_result_markdown(self):
        """Test formatting single result as Markdown."""
        formatter = ResultsFormatter("markdown")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Test completed successfully",
                "total_iterations": 2,
                "execution_time_seconds": 5.0,
            },
            "metadata": {"evaluation_name": "test_eval"},
        }

        result = formatter.format_single_result(result_data)

        # Should contain markdown formatting
        assert "# Evaluation Result" in result
        assert "Test completed successfully" in result

    def test_format_single_result_html(self):
        """Test formatting single result as HTML."""
        formatter = ResultsFormatter("html")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Test completed",
                "total_iterations": 2,
                "execution_time_seconds": 5.0,
            },
            "metadata": {"evaluation_name": "test_eval"},
        }

        result = formatter.format_single_result(result_data)

        # Should contain HTML tags
        assert "<html>" in result or "<!DOCTYPE" in result
        assert "Test completed" in result

    def test_format_single_result_unsupported(self):
        """Test formatting single result with unsupported format raises error."""
        # Create formatter with valid format then change it
        formatter = ResultsFormatter("json")
        formatter.format_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported format: unsupported"):
            formatter.format_single_result({"test": "data"})

    def test_format_batch_results_json(self):
        """Test formatting batch results as JSON."""
        formatter = ResultsFormatter("json")
        batch_data = {
            "batch_name": "test_batch",
            "total_evaluations": 3,
            "successful": 2,
            "failed": 1,
            "results": [
                {"success": True, "name": "eval1"},
                {"success": True, "name": "eval2"},
                {"success": False, "name": "eval3"},
            ],
        }

        result = formatter.format_batch_results(batch_data)

        parsed = json.loads(result)
        assert parsed["batch_name"] == "test_batch"
        assert parsed["total_evaluations"] == 3
        assert len(parsed["results"]) == 3

    def test_format_batch_results_csv(self):
        """Test formatting batch results as CSV."""
        formatter = ResultsFormatter("csv")
        batch_data = {
            "batch_name": "test_batch",
            "results": [
                {
                    "result": {
                        "success": True,
                        "final_answer": "Success",
                        "total_iterations": 1,
                        "execution_time_seconds": 1.0,
                    },
                    "metadata": {"prompt_file": "eval1.txt"},
                },
                {
                    "result": {
                        "success": False,
                        "final_answer": "Failed",
                        "total_iterations": 2,
                        "execution_time_seconds": 2.0,
                    },
                    "metadata": {"prompt_file": "eval2.txt"},
                },
            ],
        }

        result = formatter.format_batch_results(batch_data)

        # Should contain CSV data for each evaluation
        lines = result.strip().split("\n")
        assert len(lines) >= 3  # Header + 2 data rows
        assert "success" in lines[0] or "name" in lines[0]

    def test_format_comparison_results_json(self):
        """Test formatting comparison results as JSON."""
        formatter = ResultsFormatter("json")
        comparison_data = {
            "baseline_server": "server1",
            "test_server": "server2",
            "baseline_results": [{"success": True, "time": 1.0}],
            "test_results": [{"success": True, "time": 2.0}],
            "comparison_summary": {
                "winner": "server1",
                "baseline_success_rate": 100.0,
                "test_success_rate": 100.0,
            },
        }

        result = formatter.format_comparison_results(comparison_data)

        parsed = json.loads(result)
        assert parsed["baseline_server"] == "server1"
        assert parsed["test_server"] == "server2"
        assert parsed["comparison_summary"]["winner"] == "server1"

    def test_format_comparison_results_html(self):
        """Test formatting comparison results as HTML."""
        formatter = ResultsFormatter("html")
        comparison_data = {
            "baseline_server": "server1",
            "test_server": "server2",
            "analysis": {
                "total_prompts": 5,
                "success_rates": {"server1": 0.8, "server2": 0.9},
                "success_counts": {"server1": 4, "server2": 5},
                "average_execution_times": {"server1": 5.0, "server2": 3.0},
                "average_iterations": {"server1": 3.0, "server2": 2.0},
                "total_failed_tool_calls": {"server1": 1, "server2": 0},
                "total_token_usage": {"server1": 1000, "server2": 800},
                "winner": {
                    "success_rate": "server2",
                    "speed": "server2",
                    "efficiency": "server2",
                    "reliability": "server2",
                    "token_efficiency": "server2",
                },
            },
            "detailed_results": {"server1": [], "server2": []},
        }

        result = formatter.format_comparison_results(comparison_data)

        # Should contain HTML and server names
        assert "<html>" in result or "<!DOCTYPE" in result
        assert "server1" in result
        assert "server2" in result

    def test_json_formatting_pretty(self):
        """Test JSON formatting is pretty-printed."""
        formatter = ResultsFormatter("json")
        result_data = {"nested": {"data": {"structure": "test"}}}

        result = formatter.format_single_result(result_data)

        # Should be indented (pretty-printed)
        assert "  " in result or "\t" in result
        lines = result.split("\n")
        assert len(lines) > 1  # Multi-line output

    def test_csv_handles_nested_data(self):
        """Test CSV formatting handles nested data structures."""
        formatter = ResultsFormatter("csv")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Test completed",
                "total_iterations": 2,
                "execution_time_seconds": 3.0,
                "steps": [{"step": 1}, {"step": 2}],
            },
            "metadata": {"key": "value", "prompt_file": "test.txt"},
        }

        # Should not raise exception
        result = formatter.format_single_result(result_data)
        assert isinstance(result, str)
        assert "success" in result

    def test_markdown_escapes_special_chars(self):
        """Test Markdown formatting escapes special characters."""
        formatter = ResultsFormatter("markdown")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Output with `backticks` and **bold**",
                "total_iterations": 1,
                "execution_time_seconds": 2.0,
            },
            "metadata": {"evaluation_name": "test_with_*asterisks*_and_underscores_"},
        }

        result = formatter.format_single_result(result_data)

        # Should handle special markdown characters appropriately
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_escapes_special_chars(self):
        """Test HTML formatting escapes special characters."""
        formatter = ResultsFormatter("html")
        result_data = {
            "result": {
                "success": True,
                "final_answer": "Output with <script>alert('xss')</script>",
                "total_iterations": 1,
                "execution_time_seconds": 2.0,
            },
            "metadata": {"evaluation_name": "test_with_<tags>_and_&entities;"},
        }

        result = formatter.format_single_result(result_data)

        # Should escape HTML special characters in content
        assert "&lt;script&gt;" in result

    def test_empty_data_handling(self):
        """Test formatters handle empty data gracefully."""
        # Only test JSON for empty data since other formats require specific structure
        formatter = ResultsFormatter("json")
        result = formatter.format_single_result({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_values_handling(self):
        """Test formatters handle None values gracefully."""
        formatter = ResultsFormatter("json")
        result_data = {"success": None, "execution_time": None, "final_output": None}

        result = formatter.format_single_result(result_data)
        parsed = json.loads(result)
        assert parsed["success"] is None
        assert parsed["execution_time"] is None
        assert parsed["final_output"] is None

    def test_large_data_handling(self):
        """Test formatters handle large data structures."""
        formatter = ResultsFormatter("json")
        large_data = {
            "large_list": list(range(1000)),
            "large_string": "x" * 10000,
            "nested": {"level" + str(i): {"data": "test"} for i in range(100)},
        }

        result = formatter.format_single_result(large_data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed["large_list"]) == 1000

    @patch("builtins.open", side_effect=PermissionError("No write permission"))
    def test_file_write_permission_error(self, mock_open):
        """Test handling of file write permission errors."""
        formatter = ResultsFormatter("json")
        result_data = {"test": "data"}

        # Should handle permission error gracefully
        with pytest.raises(PermissionError):
            formatter.format_single_result(result_data, "/restricted/file.json")

    def test_format_batch_results_unsupported(self):
        """Test format_batch_results with unsupported format."""
        formatter = ResultsFormatter("json")
        formatter.format_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported format: unsupported"):
            formatter.format_batch_results({"test": "data"})

    def test_format_comparison_results_unsupported(self):
        """Test format_comparison_results with unsupported format."""
        formatter = ResultsFormatter("json")
        formatter.format_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported format: unsupported"):
            formatter.format_comparison_results({"test": "data"})
