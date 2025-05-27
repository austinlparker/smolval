"""Results formatting and output utilities."""

import json
import re
import time
from pathlib import Path
from typing import Any

import markdown_it
from jinja2 import Environment, FileSystemLoader, Template


class ResultsFormatter:
    """Formats evaluation results for different output formats."""

    def __init__(self, format_type: str = "json"):
        """Initialize the results formatter."""
        if format_type not in ("json", "csv", "markdown", "html"):
            raise ValueError(f"Unsupported format: {format_type}")
        self.format_type = format_type

        # Set up Jinja2 environment for template loading
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir), autoescape=True
        )

        # Add markdown filter for rendering markdown text
        md = markdown_it.MarkdownIt()
        self.jinja_env.filters["markdown"] = lambda text: (
            md.render(text) if text else ""
        )

    def format_single_result(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format a single evaluation result."""
        # Check if this is a judged result
        is_judged = "judgment" in result_data and "original_result" in result_data
        
        if self.format_type == "json":
            return self._format_json(result_data, output_file)
        elif self.format_type == "csv":
            if is_judged:
                return self._format_judged_csv(result_data, output_file)
            return self._format_single_csv(result_data, output_file)
        elif self.format_type == "markdown":
            if is_judged:
                return self._format_judged_markdown(result_data, output_file)
            return self._format_single_markdown(result_data, output_file)
        elif self.format_type == "html":
            if is_judged:
                return self._format_judged_html(result_data, output_file)
            return self._format_single_html(result_data, output_file)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")

    def format_batch_results(
        self, batch_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format batch evaluation results."""
        if self.format_type == "json":
            return self._format_json(batch_data, output_file)
        elif self.format_type == "csv":
            return self._format_batch_csv(batch_data, output_file)
        elif self.format_type == "markdown":
            return self._format_batch_markdown(batch_data, output_file)
        elif self.format_type == "html":
            return self._format_batch_html(batch_data, output_file)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")

    def format_comparison_results(
        self, comparison_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format comparison results."""
        if self.format_type == "json":
            return self._format_json(comparison_data, output_file)
        elif self.format_type == "csv":
            return self._format_comparison_csv(comparison_data, output_file)
        elif self.format_type == "markdown":
            return self._format_comparison_markdown(comparison_data, output_file)
        elif self.format_type == "html":
            return self._format_comparison_html(comparison_data, output_file)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")

    def _format_json(self, data: dict[str, Any], output_file: str | None = None) -> str:
        """Format data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        if output_file:
            with open(output_file, "w") as f:
                f.write(json_str)
        return json_str

    def _format_single_csv(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format single result as CSV."""
        result = result_data["result"]

        # Create a flat structure for CSV
        csv_data = [
            {
                "prompt_file": result_data.get("metadata", {}).get("prompt_file", ""),
                "success": result["success"],
                "final_answer": (
                    result["final_answer"]
                    .replace("\n", " ")
                    .replace("\r", "")
                ),
                "total_iterations": result["total_iterations"],
                "execution_time_seconds": result["execution_time_seconds"],
                "failed_tool_calls": result.get("failed_tool_calls", 0),
                "error": result.get("error", ""),
                "num_steps": len(result.get("steps", [])),
                "timestamp": result_data.get("metadata", {}).get(
                    "timestamp", time.time()
                ),
            }
        ]

        return self._write_csv(csv_data, output_file)

    def _format_batch_csv(
        self, batch_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format batch results as CSV."""
        csv_data = []

        for result_item in batch_data["results"]:
            result = result_item["result"]
            csv_data.append(
                {
                    "prompt_file": result_item.get("metadata", {}).get(
                        "prompt_file", ""
                    ),
                    "success": result["success"],
                    "final_answer": (
                        result["final_answer"]
                        .replace("\n", " ")
                        .replace("\r", "")
                    ),
                    "total_iterations": result["total_iterations"],
                    "execution_time_seconds": result["execution_time_seconds"],
                    "failed_tool_calls": result.get("failed_tool_calls", 0),
                    "error": result.get("error", ""),
                    "num_steps": len(result.get("steps", [])),
                    "timestamp": result_item.get("metadata", {}).get(
                        "timestamp", time.time()
                    ),
                }
            )

        return self._write_csv(csv_data, output_file)

    def _format_comparison_csv(
        self, comparison_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format comparison results as CSV."""
        csv_data = []

        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]

        baseline_results = comparison_data["detailed_results"][baseline]
        test_results = comparison_data["detailed_results"][test]

        for _i, (b_result, t_result) in enumerate(
            zip(baseline_results, test_results, strict=True)
        ):
            csv_data.append(
                {
                    "prompt_file": b_result["prompt_file"],
                    f"{baseline}_success": b_result["success"],
                    f"{baseline}_execution_time": b_result["execution_time"],
                    f"{baseline}_iterations": b_result["iterations"],
                    f"{test}_success": t_result["success"],
                    f"{test}_execution_time": t_result["execution_time"],
                    f"{test}_iterations": t_result["iterations"],
                    "winner_success": (
                        "tie"
                        if b_result["success"] == t_result["success"]
                        else (
                            baseline
                            if b_result["success"] > t_result["success"]
                            else test
                        )
                    ),
                    "winner_speed": (
                        "tie"
                        if abs(b_result["execution_time"] - t_result["execution_time"])
                        < 0.1
                        else (
                            baseline
                            if b_result["execution_time"] < t_result["execution_time"]
                            else test
                        )
                    ),
                    "winner_efficiency": (
                        "tie"
                        if b_result["iterations"] == t_result["iterations"]
                        else (
                            baseline
                            if b_result["iterations"] < t_result["iterations"]
                            else test
                        )
                    ),
                }
            )

        return self._write_csv(csv_data, output_file)

    def _write_csv(
        self, data: list[dict[str, Any]], output_file: str | None = None
    ) -> str:
        """Write data to CSV format."""
        if not data:
            return ""

        output = []

        # Write header
        fieldnames = list(data[0].keys())
        output.append(",".join(fieldnames))

        # Write rows
        for row in data:
            csv_row = []
            for field in fieldnames:
                value = str(row.get(field, ""))
                # Escape quotes and wrap in quotes if contains comma
                if "," in value or '"' in value:
                    value = '"' + value.replace('"', '""') + '"'
                csv_row.append(value)
            output.append(",".join(csv_row))

        csv_content = "\n".join(output)

        if output_file:
            with open(output_file, "w", newline="") as f:
                f.write(csv_content)

        return csv_content

    def _truncate_observation(self, observation: str, max_length: int = 1000) -> str:
        """Truncate long observations and extract key information."""
        if len(observation) <= max_length:
            return observation

        # Try to extract key information from JSON responses
        try:
            data = json.loads(observation)
            if isinstance(data, dict):
                # For Honeycomb data, extract summary info
                if "datasets" in data:
                    dataset_count = len(data["datasets"])
                    dataset_names = [
                        d.get("name", "unknown") for d in data["datasets"][:5]
                    ]
                    return f"Found {dataset_count} datasets: {', '.join(dataset_names)}{'...' if dataset_count > 5 else ''}"
                elif "events" in data:
                    event_count = len(data["events"])
                    return f"Query returned {event_count} events (data truncated for readability)"
                elif "error" in data:
                    return f"Error: {data['error']}"
                elif "message" in data:
                    return str(data["message"])
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: truncate and add indicator
        return (
            observation[:max_length]
            + f"\n\n... (truncated, {len(observation)} total characters)"
        )

    def _extract_performance_metrics(self, steps: list[dict]) -> dict[str, Any]:
        """Extract performance metrics from agent steps."""
        metrics = {
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "tools_used": set(),
            "error_patterns": [],
            "query_results": [],
        }

        for step in steps:
            if step.get("action"):
                metrics["total_tool_calls"] += 1  # type: ignore[operator]
                metrics["tools_used"].add(step["action"])  # type: ignore[attr-defined]

                observation = step.get("observation", "")
                if "error" in observation.lower() or "failed" in observation.lower():
                    metrics["failed_tool_calls"] += 1  # type: ignore[operator]
                    # Extract error message
                    try:
                        data = json.loads(observation)
                        if isinstance(data, dict) and "error" in data:
                            metrics["error_patterns"].append(data["error"])  # type: ignore[attr-defined]
                    except Exception:
                        pass
                else:
                    metrics["successful_tool_calls"] += 1  # type: ignore[operator]

                # Extract query results for visualization
                try:
                    data = json.loads(observation)
                    if isinstance(data, dict) and "events" in data:
                        metrics["query_results"].append(  # type: ignore[attr-defined]
                            {
                                "action": step["action"],
                                "event_count": len(data["events"]),
                                "iteration": step.get("iteration"),
                            }
                        )
                except Exception:
                    pass

        metrics["tools_used"] = list(metrics["tools_used"])  # type: ignore[call-overload]
        return metrics

    def _format_single_markdown(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format single result as Markdown."""
        result = result_data["result"]
        metadata = result_data.get("metadata", {})

        template = Template(
            """# Evaluation Result

## Summary
- **Success**: {{ '✅' if result.success else '❌' }}
- **Execution Time**: {{ "%.2f"|format(result.execution_time_seconds) }}s
- **Iterations**: {{ result.total_iterations }}
- **Steps**: {{ result.steps|length }}
- **Failed Tool Calls**: {{ result.failed_tool_calls or 0 }}
- **Timestamp**: {{ timestamp }}

{% if result.error %}
## Error
```
{{ result.error }}
```
{% endif %}

## Prompt
```
{{ prompt }}
```

## Final Answer
{{ result.final_answer }}

{% if result.steps %}
## Step-by-step Execution

{% for step in result.steps %}
### Step {{ loop.index }} (Iteration {{ step.iteration }})

**Thought**: {{ step.thought }}

{% if step.action %}
**Action**: {{ step.action }}

**Action Input**:
```json
{{ step.action_input|tojson(indent=2) }}
```
{% endif %}

{% if step.observation %}
**Observation**: {{ _truncate_observation(step.observation) }}
{% endif %}

---
{% endfor %}
{% endif %}

## Metadata
- **Config File**: {{ metadata.config_file or 'N/A' }}
- **Prompt File**: {{ metadata.prompt_file or 'N/A' }}
- **Duration**: {{ "%.2f"|format(metadata.duration_seconds or 0) }}s
"""
        )

        markdown_content = template.render(
            result=result,
            metadata=metadata,
            prompt=result_data.get("prompt", ""),
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(metadata.get("timestamp", time.time())),
            ),
            _truncate_observation=self._truncate_observation,
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_content)

        return markdown_content

    def _format_batch_markdown(
        self, batch_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format batch results as Markdown."""
        template = Template(
            """# Batch Evaluation Results

## Summary
- **Total Prompts**: {{ batch_data.total_prompts }}
- **Successful**: {{ batch_data.successful }} ✅
- **Failed**: {{ batch_data.failed }} ❌
- **Success Rate**: {{ "%.1f"|format(batch_data.successful / batch_data.total_prompts * 100) }}%
- **Servers**: {{ batch_data.metadata.servers|join(', ') }}
- **Timestamp**: {{ timestamp }}

## Individual Results

| Prompt File | Success | Time (s) | Iterations | Failed Tools | Error |
|-------------|---------|----------|------------|--------------|-------|
{% for result_item in batch_data.results %}
| {{ result_item.prompt_file }} | {{ '✅' if result_item.result.success else '❌' }} | {{ "%.2f"|format(result_item.result.execution_time_seconds) }} | {{ result_item.result.total_iterations }} | {{ result_item.result.failed_tool_calls or 0 }} | {{ result_item.result.error or '' }} |
{% endfor %}

## Configuration
- **Config File**: {{ batch_data.config_file }}
- **Prompts Directory**: {{ batch_data.prompts_directory }}
"""
        )

        markdown_content = template.render(
            batch_data=batch_data,
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(
                    batch_data.get("metadata", {}).get("timestamp", time.time())
                ),
            ),
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_content)

        return markdown_content

    def _format_comparison_markdown(
        self, comparison_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format comparison results as Markdown."""
        analysis = comparison_data["analysis"]
        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]

        template = Template(
            """# Server Comparison Results

## Summary
**{{ baseline }}** vs **{{ test }}**

### Overall Winners
- **Success Rate**: {{ analysis.winner.success_rate }} 🏆
- **Speed**: {{ analysis.winner.speed }} 🏆
- **Efficiency**: {{ analysis.winner.efficiency }} 🏆

### Detailed Metrics

| Metric | {{ baseline }} | {{ test }} | Winner |
|--------|{{ '-' * baseline|length }}|{{ '-' * test|length }}|--------|
| Success Rate | {{ analysis.success_counts[baseline] }}/{{ analysis.total_prompts }} ({{ "%.1f"|format(analysis.success_rates[baseline] * 100) }}%) | {{ analysis.success_counts[test] }}/{{ analysis.total_prompts }} ({{ "%.1f"|format(analysis.success_rates[test] * 100) }}%) | {{ analysis.winner.success_rate }} |
| Avg Execution Time | {{ "%.2f"|format(analysis.average_execution_times[baseline]) }}s | {{ "%.2f"|format(analysis.average_execution_times[test]) }}s | {{ analysis.winner.speed }} |
| Avg Iterations | {{ "%.1f"|format(analysis.average_iterations[baseline]) }} | {{ "%.1f"|format(analysis.average_iterations[test]) }} | {{ analysis.winner.efficiency }} |
| Failed Tool Calls | {{ analysis.total_failed_tool_calls[baseline] or 0 }} | {{ analysis.total_failed_tool_calls[test] or 0 }} | {{ baseline if (analysis.total_failed_tool_calls[baseline] or 0) <= (analysis.total_failed_tool_calls[test] or 0) else test }} |

## Per-Prompt Comparison

| Prompt | {{ baseline }} Success | {{ baseline }} Time | {{ test }} Success | {{ test }} Time | Winner |
|--------|{{ '-' * baseline|length }}-|{{ '-' * baseline|length }}--|{{ '-' * test|length }}-|{{ '-' * test|length }}--|--------|
{% for b_result, t_result in zip(comparison_data.detailed_results[baseline], comparison_data.detailed_results[test]) %}
| {{ b_result.prompt_file }} | {{ '✅' if b_result.success else '❌' }} | {{ "%.2f"|format(b_result.execution_time) }}s | {{ '✅' if t_result.success else '❌' }} | {{ "%.2f"|format(t_result.execution_time) }}s | {{ baseline if (b_result.success and b_result.execution_time <= t_result.execution_time) or (b_result.success and not t_result.success) else test }} |
{% endfor %}

## Detailed Execution Results

{% for b_result, t_result in zip(comparison_data.detailed_results[baseline], comparison_data.detailed_results[test]) %}
### {{ b_result.prompt_file }}

#### {{ baseline }} Result
- **Success**: {{ '✅' if b_result.success else '❌' }}
- **Execution Time**: {{ "%.2f"|format(b_result.execution_time) }}s
- **Iterations**: {{ b_result.iterations }}
- **Failed Tool Calls**: {{ b_result.failed_tool_calls or 0 }}
{% if b_result.error %}
- **Error**: {{ b_result.error }}
{% endif %}

**Final Answer**:
{{ b_result.final_answer or 'No final answer provided' }}

**Steps**: {{ b_result.steps }} steps completed
{% for step in b_result.detailed_steps %}
- Step {{ loop.index }}: {{ step.action or 'Thinking' }}
{% endfor %}

#### {{ test }} Result
- **Success**: {{ '✅' if t_result.success else '❌' }}
- **Execution Time**: {{ "%.2f"|format(t_result.execution_time) }}s
- **Iterations**: {{ t_result.iterations }}
- **Failed Tool Calls**: {{ t_result.failed_tool_calls or 0 }}
{% if t_result.error %}
- **Error**: {{ t_result.error }}
{% endif %}

**Final Answer**:
{{ t_result.final_answer or 'No final answer provided' }}

**Steps**: {{ t_result.steps }} steps completed
{% for step in t_result.detailed_steps %}
- Step {{ loop.index }}: {{ step.action or 'Thinking' }}
{% endfor %}

---
{% endfor %}

## Configuration
- **Config File**: {{ comparison_data.config_file }}
- **Prompts Directory**: {{ comparison_data.prompts_directory }}
- **Total Prompts**: {{ analysis.total_prompts }}
- **Timestamp**: {{ timestamp }}
"""
        )

        markdown_content = template.render(
            comparison_data=comparison_data,
            analysis=analysis,
            baseline=baseline,
            test=test,
            zip=zip,
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(
                    comparison_data.get("metadata", {}).get("timestamp", time.time())
                ),
            ),
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_content)

        return markdown_content

    def _format_single_html(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format single result as modern, interactive step-by-step HTML."""
        result = result_data["result"]
        metadata = result_data.get("metadata", {})
        metrics = self._extract_performance_metrics(result.get("steps", []))
        # Load the interactive HTML template from external file
        template = self.jinja_env.get_template("single_result.html")

        html_content = template.render(
            result=result,
            metadata=metadata,
            metrics=metrics,
            prompt=result_data.get("prompt", ""),
            total_tokens=result.get("total_tokens", 0),
            success=result.get("success", False),
            error=result.get("error"),
            completion_time=result.get("execution_time_seconds", 0),
            mcp_servers=result.get("mcp_servers", []),
            steps=result.get("steps", []),
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(metadata.get("timestamp", time.time())),
            ),
            _truncate_observation=self._truncate_observation,
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(html_content)

        return html_content

    def _format_batch_html(
        self, batch_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format batch results as interactive HTML with step-by-step visualization."""
        # Load the interactive batch template from external file
        template = self.jinja_env.get_template("batch_result.html")

        html_content = template.render(
            batch_data=batch_data,
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(
                    batch_data.get("metadata", {}).get("timestamp", time.time())
                ),
            ),
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(html_content)

        return html_content

    def _format_comparison_html(
        self, comparison_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format comparison results as modern, interactive HTML."""
        analysis = comparison_data["analysis"]
        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]

        # Load the interactive comparison template from external file
        template = self.jinja_env.get_template("comparison_result.html")

        html_content = template.render(
            comparison_data=comparison_data,
            analysis=analysis,
            baseline=baseline,
            test=test,
            zip=zip,
            timestamp=time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(
                    comparison_data.get("metadata", {}).get("timestamp", time.time())
                ),
            ),
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(html_content)

        return html_content

    def _markdown_to_html(self, markdown_content: str, title: str) -> str:
        """Convert markdown to compact, information-dense HTML."""

        # Process markdown with better HTML structure
        lines = markdown_content.split("\n")
        html_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Headers with collapsible sections
            if line.startswith("### "):
                header_text = line[4:]
                html_lines.append(
                    f'<details class="subsection"><summary class="h3">{header_text}</summary><div class="content">'
                )
            elif line.startswith("## "):
                # Close any open details first
                if html_lines and "<details" in html_lines[-1]:
                    html_lines.append("</div></details>")
                header_text = line[3:]
                html_lines.append(f"<h2>{header_text}</h2>")
            elif line.startswith("# "):
                header_text = line[2:]
                html_lines.append(f"<h1>{header_text}</h1>")
            elif line.startswith("- **"):
                # Summary list items
                match = re.match(r"- \*\*(.*?)\*\*: (.*)", line)
                if match:
                    key, value = match.groups()
                    html_lines.append(
                        f'<div class="summary-item"><span class="key">{key}:</span> <span class="value">{value}</span></div>'
                    )
                else:
                    html_lines.append(f'<div class="list-item">{line[2:]}</div>')
            elif line.startswith("- "):
                html_lines.append(f'<div class="list-item">{line[2:]}</div>')
            elif line.startswith("```"):
                # Skip code block markers, we'll handle content
                continue
            elif line.startswith("**") and line.endswith("**"):
                content = line[2:-2]
                html_lines.append(f'<div class="section-label">{content}</div>')
            else:
                # Regular content with word wrapping
                html_lines.append(f'<div class="content-line">{line}</div>')

        # Close any remaining open details
        if html_lines and any("<details" in line for line in html_lines):
            html_lines.append("</div></details>")

        html_content = "\n".join(html_lines)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.4;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
        }}

        .container {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        h1 {{
            font-size: 20px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 16px;
            border-bottom: 2px solid #e1e5e9;
            padding-bottom: 8px;
        }}

        h2 {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
            margin: 20px 0 12px 0;
            padding: 8px 12px;
            background: #f7fafc;
            border-left: 4px solid #3182ce;
            border-radius: 4px;
        }}

        .subsection {{
            margin: 12px 0;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }}

        .subsection summary {{
            padding: 8px 12px;
            background: #f8fafc;
            cursor: pointer;
            font-weight: 500;
            color: #4a5568;
            border-radius: 6px 6px 0 0;
            user-select: none;
        }}

        .subsection summary:hover {{
            background: #edf2f7;
        }}

        .subsection .content {{
            padding: 12px;
            background: white;
        }}

        .summary-item {{
            display: flex;
            padding: 4px 0;
            border-bottom: 1px solid #f1f5f9;
        }}

        .summary-item:last-child {{
            border-bottom: none;
        }}

        .summary-item .key {{
            font-weight: 500;
            color: #2d3748;
            min-width: 140px;
            flex-shrink: 0;
        }}

        .summary-item .value {{
            color: #4a5568;
            word-break: break-word;
        }}

        .list-item {{
            padding: 3px 0;
            margin-left: 16px;
            position: relative;
            color: #4a5568;
            word-wrap: break-word;
        }}

        .list-item:before {{
            content: "•";
            position: absolute;
            left: -12px;
            color: #a0aec0;
        }}

        .content-line {{
            margin: 6px 0;
            color: #4a5568;
            word-wrap: break-word;
            white-space: pre-wrap;
        }}

        .section-label {{
            font-weight: 600;
            color: #2d3748;
            margin: 12px 0 6px 0;
            padding: 6px 0;
            border-bottom: 1px solid #e2e8f0;
        }}

        pre {{
            background: #f7fafc;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #e2e8f0;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .status-success {{ color: #22c55e; font-weight: 600; }}
        .status-error {{ color: #ef4444; font-weight: 600; }}
        /* Make content more compact on smaller screens */
        @media (max-width: 768px) {{
            body {{ padding: 12px; font-size: 13px; }}
            .container {{ padding: 16px; }}
            h1 {{ font-size: 18px; }}
            h2 {{ font-size: 15px; }}
            .summary-item {{ flex-direction: column; }}
            .summary-item .key {{ min-width: auto; margin-bottom: 2px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>

    <script>
        // Auto-expand first few sections by default
        document.addEventListener('DOMContentLoaded', function() {{
            const details = document.querySelectorAll('details');
            details.forEach((detail, index) => {{
                if (index < 2) {{ // Expand first 2 sections
                    detail.open = true;
                }}
            }});
        }});
    </script>
</body>
</html>"""

    def _format_judged_csv(
        self, judged_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format judged result as CSV."""
        original_result = judged_data["original_result"]
        judgment = judged_data["judgment"]
        result = original_result["result"]

        # Create a flat structure for CSV including judgment scores
        csv_data = [
            {
                "prompt_file": original_result.get("metadata", {}).get("prompt_file", ""),
                "success": result["success"],
                "final_answer": (
                    result["final_answer"]
                    .replace("\n", " ")
                    .replace("\r", "")
                ),
                "total_iterations": result["total_iterations"],
                "execution_time_seconds": result["execution_time_seconds"],
                "failed_tool_calls": result.get("failed_tool_calls", 0),
                "error": result.get("error", ""),
                "num_steps": len(result.get("steps", [])),
                "timestamp": original_result.get("metadata", {}).get(
                    "timestamp", time.time()
                ),
                # Judgment data
                "overall_score": judgment["overall_score"],
                "judgment_summary": judgment["summary"].replace("\n", " "),
                "strengths": "; ".join(judgment.get("strengths", [])),
                "weaknesses": "; ".join(judgment.get("weaknesses", [])),
                **{
                    f"{score['criterion']}_score": score["score"]
                    for score in judgment["scores"]
                },
            }
        ]

        return self._write_csv(csv_data, output_file)

    def _format_judged_markdown(
        self, judged_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format judged result as Markdown."""
        original_result = judged_data["original_result"]
        judgment = judged_data["judgment"]
        result = original_result["result"]
        metadata = original_result.get("metadata", {})

        # First generate the standard result markdown
        standard_markdown = self._format_single_markdown(original_result, None)
        
        # Add judgment section
        judgment_section = f"""

## 🎯 LLM-as-Judge Evaluation

### Overall Assessment
**Quality Score: {judgment['overall_score']:.2f}/1.0**

{judgment['summary']}

### Detailed Scores

| Criterion | Score | Reasoning |
|-----------|--------|-----------|
"""
        
        for score in judgment["scores"]:
            reasoning = score["reasoning"][:100] + "..." if len(score["reasoning"]) > 100 else score["reasoning"]
            judgment_section += f"| {score['criterion'].replace('_', ' ').title()} | {score['score']:.2f} | {reasoning} |\n"

        if judgment.get("strengths"):
            judgment_section += f"""
### ✅ Strengths
{chr(10).join(f'- {strength}' for strength in judgment['strengths'])}
"""

        if judgment.get("weaknesses"):
            judgment_section += f"""
### ⚠️ Areas for Improvement
{chr(10).join(f'- {weakness}' for weakness in judgment['weaknesses'])}
"""

        if judgment.get("suggestions"):
            judgment_section += f"""
### 💡 Suggestions
{chr(10).join(f'- {suggestion}' for suggestion in judgment['suggestions'])}
"""

        # Combine standard and judgment sections
        full_markdown = standard_markdown + judgment_section

        if output_file:
            with open(output_file, "w") as f:
                f.write(full_markdown)

        return full_markdown

    def _format_judged_html(
        self, judged_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format judged result as HTML."""
        original_result = judged_data["original_result"]
        judgment = judged_data["judgment"]
        
        # Generate standard HTML first
        standard_html = self._format_single_html(original_result, None)
        
        # Extract the content between <body> tags to inject judgment
        body_match = re.search(r'<div class="container">(.*?)</div>\s*<script>', standard_html, re.DOTALL)
        
        if body_match:
            content = body_match.group(1)
            
            # Add judgment section
            judgment_html = f"""
        <h2>🎯 LLM-as-Judge Evaluation</h2>
        
        <div class="summary-item">
            <span class="key">Overall Quality Score:</span> 
            <span class="value" style="font-weight: bold; color: {'#22c55e' if judgment['overall_score'] > 0.7 else '#f59e0b' if judgment['overall_score'] > 0.4 else '#ef4444'};">
                {judgment['overall_score']:.2f}/1.0
            </span>
        </div>
        
        <div class="content-line" style="margin: 12px 0; padding: 12px; background: #f8fafc; border-radius: 6px;">
            {judgment['summary']}
        </div>
        
        <h3>Detailed Scores</h3>
        <div style="display: grid; gap: 8px; margin: 12px 0;">
"""
            
            for score in judgment["scores"]:
                score_color = "#22c55e" if score["score"] > 0.7 else "#f59e0b" if score["score"] > 0.4 else "#ef4444"
                judgment_html += f"""
            <details class="subsection">
                <summary class="h3">
                    {score['criterion'].replace('_', ' ').title()}: 
                    <span style="color: {score_color}; font-weight: bold;">{score['score']:.2f}</span>
                </summary>
                <div class="content">
                    <div class="content-line">{score['reasoning']}</div>
                </div>
            </details>
"""
            
            judgment_html += "</div>"
            
            if judgment.get("strengths"):
                judgment_html += """
        <h3>✅ Strengths</h3>
        <div class="content">
"""
                for strength in judgment["strengths"]:
                    judgment_html += f'            <div class="list-item">{strength}</div>\n'
                judgment_html += "        </div>"
            
            if judgment.get("weaknesses"):
                judgment_html += """
        <h3>⚠️ Areas for Improvement</h3>
        <div class="content">
"""
                for weakness in judgment["weaknesses"]:
                    judgment_html += f'            <div class="list-item">{weakness}</div>\n'
                judgment_html += "        </div>"
            
            # Replace the content in the original HTML
            full_html = standard_html.replace(
                body_match.group(1),
                content + judgment_html
            )
        else:
            # Fallback if parsing fails
            full_html = standard_html
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(full_html)

        return full_html
