"""Tests for output_manager module."""

from smolval.models import AgentResult, AgentStep, ExecutionMetadata
from smolval.output_manager import OutputManager


class TestOutputManager:
    """Test OutputManager class."""

    def test_format_result_markdown(self):
        """Test formatting result as markdown."""
        from datetime import datetime

        manager = OutputManager()

        # Create a mock result with new structure
        metadata = ExecutionMetadata(
            session_id="test-session",
            model_used="claude-3-sonnet",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
        )

        result = AgentResult(
            success=True,
            final_answer="Test answer",
            steps=[
                AgentStep(
                    step_id="step_1",
                    iteration=1,
                    step_type="text_response",
                    content="Test content",
                    tool_name="test_tool",
                )
            ],
            total_iterations=1,
            execution_time_seconds=1.5,
            metadata=metadata,
        )

        output = manager.format_result(result, "markdown")

        assert "# Evaluation Result" in output
        assert "✅ Success" in output
        assert "Test answer" in output
        assert "Test content" in output

    def test_format_result_json(self):
        """Test formatting result as JSON."""
        from datetime import datetime

        manager = OutputManager()

        metadata = ExecutionMetadata(
            session_id="test-session",
            model_used="claude-3-sonnet",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
        )

        result = AgentResult(
            success=False,
            final_answer="",
            steps=[],
            total_iterations=0,
            execution_time_seconds=0.5,
            error_message="Test error",
            metadata=metadata,
        )

        output = manager.format_result(result, "json")

        assert '"success": false' in output
        assert '"error_message": "Test error"' in output
        assert '"failed_tool_calls": 0' in output

    def test_format_result_html(self):
        """Test formatting result as HTML."""
        from datetime import datetime

        manager = OutputManager()

        metadata = ExecutionMetadata(
            session_id="test-session",
            model_used="claude-3-sonnet",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
        )

        result = AgentResult(
            success=True,
            final_answer="HTML test",
            steps=[],
            total_iterations=0,
            execution_time_seconds=2.0,
            metadata=metadata,
        )

        output = manager.format_result(result, "html")

        assert "<html" in output
        assert "HTML test" in output
        # Check for success indicator in template (may be emoji or icon)
        assert "✅" in output or "success" in output.lower()

    def test_format_result_csv(self):
        """Test formatting result as CSV."""
        from datetime import datetime

        manager = OutputManager()

        metadata = ExecutionMetadata(
            session_id="test-session",
            model_used="claude-3-sonnet",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
        )

        result = AgentResult(
            success=True,
            final_answer="CSV test answer",
            steps=[],
            total_iterations=3,
            execution_time_seconds=4.5,
            metadata=metadata,
        )

        output = manager.format_result(result, "csv")

        assert "success,execution_time_seconds" in output
        # Note: CSV format has been updated with new fields
        assert "True,4.5,3,0,0,15" in output  # 15 is length of "CSV test answer"
