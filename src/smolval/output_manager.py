"""Output management for organizing evaluation results."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .results import ResultsFormatter


class OutputManager:
    """Manages output organization and generation."""

    def __init__(self, base_output_dir: str = "results"):
        """Initialize the output manager."""
        self.base_output_dir = Path(base_output_dir)
        self.current_run_dir: Path | None = None

    def create_run_directory(self, run_name: str | None = None) -> Path:
        """Create a timestamped directory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_name:
            # Sanitize run name for filesystem
            safe_run_name = "".join(c for c in run_name if c.isalnum() or c in ('-', '_')).strip()
            dir_name = f"{timestamp}_{safe_run_name}"
        else:
            dir_name = timestamp

        self.current_run_dir = self.base_output_dir / dir_name
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        return self.current_run_dir

    def get_run_directory(self) -> Path:
        """Get the current run directory, creating one if needed."""
        if self.current_run_dir is None:
            return self.create_run_directory()
        return self.current_run_dir

    def write_evaluation_results(
        self, 
        result_data: dict[str, Any], 
        eval_name: str | None = None
    ) -> dict[str, str]:
        """Write evaluation results in multiple formats."""
        run_dir = self.get_run_directory()
        
        # Generate base filename
        if eval_name:
            safe_eval_name = "".join(c for c in eval_name if c.isalnum() or c in ('-', '_')).strip()
            base_name = f"eval_{safe_eval_name}"
        else:
            base_name = f"eval_{int(time.time())}"

        # Prepare clean result data for markdown/html (without LLM metadata)
        clean_result_data = self._prepare_clean_result_data(result_data)
        
        output_files = {}
        
        # Write JSON with full metadata
        json_formatter = ResultsFormatter("json")
        json_file = run_dir / f"{base_name}.json"
        json_formatter.format_single_result(result_data, str(json_file))
        output_files["json"] = str(json_file)
        
        # Write markdown with clean data
        markdown_formatter = ResultsFormatter("markdown")
        markdown_file = run_dir / f"{base_name}.md"
        markdown_formatter.format_single_result(clean_result_data, str(markdown_file))
        output_files["markdown"] = str(markdown_file)
        
        # Write HTML with clean data
        html_formatter = ResultsFormatter("html")
        html_file = run_dir / f"{base_name}.html"
        html_formatter.format_single_result(clean_result_data, str(html_file))
        output_files["html"] = str(html_file)
        
        return output_files

    def write_batch_results(
        self, 
        batch_data: dict[str, Any], 
        batch_name: str | None = None
    ) -> dict[str, str]:
        """Write batch results in multiple formats."""
        run_dir = self.get_run_directory()
        
        # Generate base filename
        if batch_name:
            safe_batch_name = "".join(c for c in batch_name if c.isalnum() or c in ('-', '_')).strip()
            base_name = f"batch_{safe_batch_name}"
        else:
            base_name = f"batch_{int(time.time())}"

        # Prepare clean batch data for markdown/html
        clean_batch_data = self._prepare_clean_batch_data(batch_data)
        
        output_files = {}
        
        # Write JSON with full metadata
        json_formatter = ResultsFormatter("json")
        json_file = run_dir / f"{base_name}.json"
        json_formatter.format_batch_results(batch_data, str(json_file))
        output_files["json"] = str(json_file)
        
        # Write markdown with clean data
        markdown_formatter = ResultsFormatter("markdown")
        markdown_file = run_dir / f"{base_name}.md"
        markdown_formatter.format_batch_results(clean_batch_data, str(markdown_file))
        output_files["markdown"] = str(markdown_file)
        
        # Write HTML with clean data
        html_formatter = ResultsFormatter("html")
        html_file = run_dir / f"{base_name}.html"
        html_formatter.format_batch_results(clean_batch_data, str(html_file))
        output_files["html"] = str(html_file)
        
        return output_files

    def write_comparison_results(
        self, 
        comparison_data: dict[str, Any], 
        comparison_name: str | None = None
    ) -> dict[str, str]:
        """Write comparison results in multiple formats."""
        run_dir = self.get_run_directory()
        
        # Generate base filename
        if comparison_name:
            safe_comparison_name = "".join(c for c in comparison_name if c.isalnum() or c in ('-', '_')).strip()
            base_name = f"comparison_{safe_comparison_name}"
        else:
            base_name = f"comparison_{int(time.time())}"

        # Comparison data doesn't need cleaning as it doesn't contain LLM metadata
        
        output_files = {}
        
        # Write all formats
        for format_type in ["json", "markdown", "html"]:
            formatter = ResultsFormatter(format_type)
            file_path = run_dir / f"{base_name}.{format_type}"
            formatter.format_comparison_results(comparison_data, str(file_path))
            output_files[format_type] = str(file_path)
        
        return output_files

    def _prepare_clean_result_data(self, result_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare clean result data without LLM metadata for markdown/html."""
        clean_data = result_data.copy()
        
        # Clean steps to remove LLM response metadata
        if "result" in clean_data and "steps" in clean_data["result"]:
            clean_steps = []
            for step in clean_data["result"]["steps"]:
                clean_step = {k: v for k, v in step.items() if k != "llm_response"}
                clean_steps.append(clean_step)
            clean_data["result"]["steps"] = clean_steps
        
        # Remove token usage from main result for markdown/html
        if "result" in clean_data and "token_usage" in clean_data["result"]:
            clean_data["result"] = {k: v for k, v in clean_data["result"].items() if k != "token_usage"}
        
        # Remove llm_responses from markdown/html output (keep only in JSON)
        if "llm_responses" in clean_data:
            del clean_data["llm_responses"]
        
        return clean_data

    def _prepare_clean_batch_data(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare clean batch data without LLM metadata for markdown/html."""
        clean_data = batch_data.copy()
        
        # Clean each result in the batch
        if "results" in clean_data:
            clean_results = []
            for result_item in clean_data["results"]:
                clean_result_item = result_item.copy()
                # Clean the result data using the same method
                clean_result_item = self._prepare_clean_result_data(clean_result_item)
                clean_results.append(clean_result_item)
            clean_data["results"] = clean_results
        
        return clean_data