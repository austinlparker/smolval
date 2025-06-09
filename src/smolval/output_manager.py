"""Output management for organizing evaluation results."""

from .models import AgentResult
from .results import ResultsFormatter


class OutputManager:
    """Manages output formatting for evaluation results."""

    def format_result(self, result: AgentResult, format_type: str = "markdown") -> str:
        """Format an agent result in the specified format."""
        formatter = ResultsFormatter(format_type)

        # Convert AgentResult to dictionary format
        result_dict = {
            "success": result.success,
            "final_answer": result.final_answer,
            "steps": [step.model_dump() for step in result.steps],
            "total_iterations": result.total_iterations,
            "execution_time_seconds": result.execution_time_seconds,
            "failed_tool_calls": result.failed_tool_calls,
            "successful_tool_calls": result.successful_tool_calls,
            "error_message": result.error_message,
            "metadata": result.metadata.model_dump(),
        }

        return formatter.format_result(result_dict)
