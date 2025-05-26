"""Tests for CLI module."""

from smolval.cli import _analyze_comparison


class TestCLI:
    """Test CLI functions."""

    def test_analyze_comparison_basic(self):
        """Test basic comparison analysis."""
        baseline_result = {
            "success": True,
            "iterations": 3,
            "execution_time": 5.0,
            "tool_calls_successful": 2,
            "tool_calls_failed": 0,
            "total_tokens": 100,
            "steps": [
                {
                    "step": 1,
                    "thought": "Test thought",
                    "action": {"tool": "test", "arguments": {}},
                    "observation": "Test observation",
                }
            ],
            "final_output": "Test output",
        }

        test_result = {
            "success": True,
            "iterations": 2,
            "execution_time": 3.0,
            "tool_calls_successful": 2,
            "tool_calls_failed": 1,
            "total_tokens": 80,
            "steps": [
                {
                    "step": 1,
                    "thought": "Test thought",
                    "action": {"tool": "test", "arguments": {}},
                    "observation": "Test observation",
                }
            ],
            "final_output": "Test output",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        assert analysis["success_rates"]["baseline"] == 1.0
        assert analysis["success_rates"]["test"] == 1.0
        assert analysis["total_prompts"] == 1
        assert "winner" in analysis
        assert "average_execution_times" in analysis
        assert "total_token_usage" in analysis

    def test_analyze_comparison_with_failures(self):
        """Test comparison analysis with failures."""
        baseline_result = {
            "success": False,
            "iterations": 5,
            "execution_time": 10.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 2,
            "total_tokens": 200,
            "steps": [],
            "final_output": "Failed",
        }

        test_result = {
            "success": True,
            "iterations": 2,
            "execution_time": 3.0,
            "tool_calls_successful": 3,
            "tool_calls_failed": 0,
            "total_tokens": 80,
            "steps": [],
            "final_output": "Success",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        assert analysis["success_rates"]["baseline"] == 0.0
        assert analysis["success_rates"]["test"] == 1.0
        assert analysis["total_prompts"] == 1

    def test_analyze_comparison_edge_cases(self):
        """Test comparison analysis with edge cases."""
        # Test with zero tokens
        baseline_result = {
            "success": True,
            "iterations": 1,
            "execution_time": 1.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 0,
            "total_tokens": 0,
            "steps": [],
            "final_output": "Success",
        }

        test_result = {
            "success": True,
            "iterations": 1,
            "execution_time": 1.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 0,
            "total_tokens": 100,
            "steps": [],
            "final_output": "Success",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        # Should handle division by zero gracefully
        assert "total_token_usage" in analysis
        assert "baseline" in analysis["total_token_usage"]
        assert "test" in analysis["total_token_usage"]
