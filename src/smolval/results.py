"""Results formatting and output utilities."""

import json
import re
import time
from typing import Any

from jinja2 import Template


class ResultsFormatter:
    """Formats evaluation results for different output formats."""

    def __init__(self, format_type: str = "json"):
        """Initialize the results formatter."""
        if format_type not in ("json", "csv", "markdown", "html"):
            raise ValueError(f"Unsupported format: {format_type}")
        self.format_type = format_type

    def format_single_result(self, result_data: dict[str, Any], output_file: str = None) -> str:
        """Format a single evaluation result."""
        if self.format_type == "json":
            return self._format_json(result_data, output_file)
        elif self.format_type == "csv":
            return self._format_single_csv(result_data, output_file)
        elif self.format_type == "markdown":
            return self._format_single_markdown(result_data, output_file)
        elif self.format_type == "html":
            return self._format_single_html(result_data, output_file)

    def format_batch_results(self, batch_data: dict[str, Any], output_file: str = None) -> str:
        """Format batch evaluation results."""
        if self.format_type == "json":
            return self._format_json(batch_data, output_file)
        elif self.format_type == "csv":
            return self._format_batch_csv(batch_data, output_file)
        elif self.format_type == "markdown":
            return self._format_batch_markdown(batch_data, output_file)
        elif self.format_type == "html":
            return self._format_batch_html(batch_data, output_file)

    def format_comparison_results(self, comparison_data: dict[str, Any], output_file: str = None) -> str:
        """Format comparison results."""
        if self.format_type == "json":
            return self._format_json(comparison_data, output_file)
        elif self.format_type == "csv":
            return self._format_comparison_csv(comparison_data, output_file)
        elif self.format_type == "markdown":
            return self._format_comparison_markdown(comparison_data, output_file)
        elif self.format_type == "html":
            return self._format_comparison_html(comparison_data, output_file)

    def _format_json(self, data: dict[str, Any], output_file: str = None) -> str:
        """Format data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
        return json_str

    def _format_single_csv(self, result_data: dict[str, Any], output_file: str = None) -> str:
        """Format single result as CSV."""
        result = result_data["result"]

        # Create a flat structure for CSV
        csv_data = [{
            "prompt_file": result_data.get("metadata", {}).get("prompt_file", ""),
            "success": result["success"],
            "final_answer": result["final_answer"].replace('\n', ' ').replace('\r', ''),
            "total_iterations": result["total_iterations"],
            "execution_time_seconds": result["execution_time_seconds"],
            "failed_tool_calls": result.get("failed_tool_calls", 0),
            "error": result.get("error", ""),
            "num_steps": len(result.get("steps", [])),
            "timestamp": result_data.get("metadata", {}).get("timestamp", time.time())
        }]

        return self._write_csv(csv_data, output_file)

    def _format_batch_csv(self, batch_data: dict[str, Any], output_file: str = None) -> str:
        """Format batch results as CSV."""
        csv_data = []

        for result_item in batch_data["results"]:
            result = result_item["result"]
            csv_data.append({
                "prompt_file": result_item["prompt_file"],
                "success": result["success"],
                "final_answer": result["final_answer"].replace('\n', ' ').replace('\r', ''),
                "total_iterations": result["total_iterations"],
                "execution_time_seconds": result["execution_time_seconds"],
                "failed_tool_calls": result.get("failed_tool_calls", 0),
                "error": result.get("error", ""),
                "num_steps": len(result.get("steps", [])),
                "timestamp": result_item.get("metadata", {}).get("timestamp", time.time())
            })

        return self._write_csv(csv_data, output_file)

    def _format_comparison_csv(self, comparison_data: dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as CSV."""
        csv_data = []

        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]

        baseline_results = comparison_data["detailed_results"][baseline]
        test_results = comparison_data["detailed_results"][test]

        for i, (b_result, t_result) in enumerate(zip(baseline_results, test_results, strict=False)):
            csv_data.append({
                "prompt_file": b_result["prompt_file"],
                f"{baseline}_success": b_result["success"],
                f"{baseline}_execution_time": b_result["execution_time"],
                f"{baseline}_iterations": b_result["iterations"],
                f"{test}_success": t_result["success"],
                f"{test}_execution_time": t_result["execution_time"],
                f"{test}_iterations": t_result["iterations"],
                "winner_success": baseline if b_result["success"] >= t_result["success"] else test,
                "winner_speed": baseline if b_result["execution_time"] <= t_result["execution_time"] else test,
                "winner_efficiency": baseline if b_result["iterations"] <= t_result["iterations"] else test
            })

        return self._write_csv(csv_data, output_file)

    def _write_csv(self, data: list[dict[str, Any]], output_file: str = None) -> str:
        """Write data to CSV format."""
        if not data:
            return ""

        output = []

        # Write header
        fieldnames = list(data[0].keys())
        output.append(','.join(fieldnames))

        # Write rows
        for row in data:
            csv_row = []
            for field in fieldnames:
                value = str(row.get(field, ""))
                # Escape quotes and wrap in quotes if contains comma
                if ',' in value or '"' in value:
                    value = '"' + value.replace('"', '""') + '"'
                csv_row.append(value)
            output.append(','.join(csv_row))

        csv_content = '\n'.join(output)

        if output_file:
            with open(output_file, 'w', newline='') as f:
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
                    dataset_names = [d.get("name", "unknown") for d in data["datasets"][:5]]
                    return f"Found {dataset_count} datasets: {', '.join(dataset_names)}{'...' if dataset_count > 5 else ''}"
                elif "events" in data:
                    event_count = len(data["events"])
                    return f"Query returned {event_count} events (data truncated for readability)"
                elif "error" in data:
                    return f"Error: {data['error']}"
                elif "message" in data:
                    return data["message"]
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback: truncate and add indicator
        return observation[:max_length] + f"\n\n... (truncated, {len(observation)} total characters)"

    def _extract_performance_metrics(self, steps: list[dict]) -> dict[str, Any]:
        """Extract performance metrics from agent steps."""
        metrics = {
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "tools_used": set(),
            "error_patterns": [],
            "query_results": []
        }
        
        for step in steps:
            if step.get("action"):
                metrics["total_tool_calls"] += 1
                metrics["tools_used"].add(step["action"])
                
                observation = step.get("observation", "")
                if "error" in observation.lower() or "failed" in observation.lower():
                    metrics["failed_tool_calls"] += 1
                    # Extract error message
                    try:
                        data = json.loads(observation)
                        if isinstance(data, dict) and "error" in data:
                            metrics["error_patterns"].append(data["error"])
                    except:
                        pass
                else:
                    metrics["successful_tool_calls"] += 1
                
                # Extract query results for visualization
                try:
                    data = json.loads(observation)
                    if isinstance(data, dict) and "events" in data:
                        metrics["query_results"].append({
                            "action": step["action"],
                            "event_count": len(data["events"]),
                            "iteration": step.get("iteration")
                        })
                except:
                    pass
        
        metrics["tools_used"] = list(metrics["tools_used"])
        return metrics

    def _format_single_markdown(self, result_data: dict[str, Any], output_file: str = None) -> str:
        """Format single result as Markdown."""
        result = result_data["result"]
        metadata = result_data.get("metadata", {})

        template = Template("""# Evaluation Result

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
""")

        markdown_content = template.render(
            result=result,
            metadata=metadata,
            prompt=result_data.get("prompt", ""),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metadata.get("timestamp", time.time()))),
            _truncate_observation=self._truncate_observation
        )

        if output_file:
            with open(output_file, 'w') as f:
                f.write(markdown_content)

        return markdown_content

    def _format_batch_markdown(self, batch_data: dict[str, Any], output_file: str = None) -> str:
        """Format batch results as Markdown."""
        template = Template("""# Batch Evaluation Results

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
""")

        markdown_content = template.render(
            batch_data=batch_data,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(batch_data.get("metadata", {}).get("timestamp", time.time())))
        )

        if output_file:
            with open(output_file, 'w') as f:
                f.write(markdown_content)

        return markdown_content

    def _format_comparison_markdown(self, comparison_data: dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as Markdown."""
        analysis = comparison_data["analysis"]
        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]

        template = Template("""# Server Comparison Results

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
""")

        markdown_content = template.render(
            comparison_data=comparison_data,
            analysis=analysis,
            baseline=baseline,
            test=test,
            zip=zip,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(comparison_data.get("metadata", {}).get("timestamp", time.time())))
        )

        if output_file:
            with open(output_file, 'w') as f:
                f.write(markdown_content)

        return markdown_content

    def _format_single_html(self, result_data: dict[str, Any], output_file: str = None) -> str:
        """Format single result as modern, interactive HTML."""
        result = result_data["result"]
        metadata = result_data.get("metadata", {})
        metrics = self._extract_performance_metrics(result.get("steps", []))
        
        # Create modern HTML template
        template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ 'Evaluation Result: ' + (metadata.prompt_file or 'Unknown') }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #2563eb;
            --primary-light: #dbeafe;
            --success: #059669;
            --success-light: #d1fae5;
            --error: #dc2626;
            --error-light: #fee2e2;
            --warning: #d97706;
            --warning-light: #fef3c7;
            --neutral-50: #fafafa;
            --neutral-100: #f5f5f5;
            --neutral-200: #e5e5e5;
            --neutral-300: #d4d4d4;
            --neutral-700: #404040;
            --neutral-800: #262626;
            --neutral-900: #171717;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: var(--neutral-800);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, #1e40af 100%);
            color: white;
            padding: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            transform: translate(50%, -50%);
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .header .subtitle {
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            padding: 2rem;
            background: var(--neutral-50);
            border-bottom: 1px solid var(--neutral-200);
        }
        
        .status-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .status-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }
        
        .status-card .label {
            font-size: 0.875rem;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .status-card .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--neutral-900);
        }
        
        .status-card.success .value { color: var(--success); }
        .status-card.error .value { color: var(--error); }
        .status-card.warning .value { color: var(--warning); }
        
        .main-content {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--neutral-900);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .section-title::before {
            content: '';
            width: 4px;
            height: 1.5rem;
            background: var(--primary);
            border-radius: 2px;
        }
        
        .card {
            background: white;
            border: 1px solid var(--neutral-200);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        .prompt-card {
            background: var(--neutral-50);
            border-left: 4px solid var(--primary);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        .prompt-card .prompt-text {
            font-style: italic;
            color: var(--neutral-700);
            white-space: pre-wrap;
        }
        
        .final-answer {
            background: var(--success-light);
            border: 1px solid var(--success);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .final-answer h3 {
            color: var(--success);
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }
        
        .answer-content {
            white-space: pre-wrap;
            line-height: 1.7;
        }
        
        .steps-container {
            space-y: 1rem;
        }
        
        .step {
            border: 1px solid var(--neutral-200);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .step.expanded {
            box-shadow: var(--shadow);
        }
        
        .step-header {
            background: var(--neutral-100);
            padding: 1rem 1.5rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            transition: background-color 0.2s ease;
        }
        
        .step-header:hover {
            background: var(--neutral-200);
        }
        
        .step-title {
            font-weight: 600;
            color: var(--neutral-900);
        }
        
        .step-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--neutral-700);
        }
        
        .expand-icon {
            transition: transform 0.3s ease;
            color: var(--neutral-700);
        }
        
        .step.expanded .expand-icon {
            transform: rotate(180deg);
        }
        
        .step-content {
            padding: 0 1.5rem;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease, padding 0.3s ease;
        }
        
        .step.expanded .step-content {
            max-height: 1000px;
            padding: 0 1.5rem 1.5rem 1.5rem;
        }
        
        .step-section {
            margin-bottom: 1rem;
        }
        
        .step-section h4 {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .step-section .content {
            background: var(--neutral-50);
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            border: 1px solid var(--neutral-200);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        .metric-card h4 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--neutral-900);
            margin-bottom: 1rem;
        }
        
        .metric-list {
            list-style: none;
        }
        
        .metric-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--neutral-200);
            display: flex;
            justify-content: space-between;
        }
        
        .metric-list li:last-child {
            border-bottom: none;
        }
        
        .progress-bar {
            background: var(--neutral-200);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--success);
            border-radius: 8px;
            transition: width 0.3s ease;
        }
        
        .progress-fill.error {
            background: var(--error);
        }
        
        .error-card {
            background: var(--error-light);
            border: 1px solid var(--error);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .error-card h3 {
            color: var(--error);
            margin-bottom: 1rem;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 1rem 0.5rem;
            }
            
            .header {
                padding: 1.5rem;
            }
            
            .header h1 {
                font-size: 1.5rem;
            }
            
            .status-bar {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                padding: 1rem;
                gap: 0.75rem;
            }
            
            .main-content {
                padding: 1rem;
            }
            
            .step-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Evaluation Report</h1>
            <div class="subtitle">{{ metadata.prompt_file or 'Unknown Prompt' }} • {{ timestamp }}</div>
        </div>
        
        <div class="status-bar">
            <div class="status-card {{ 'success' if result.success else 'error' }}">
                <div class="label">Status</div>
                <div class="value">{{ '✅ Success' if result.success else '❌ Failed' }}</div>
            </div>
            <div class="status-card">
                <div class="label">Execution Time</div>
                <div class="value">{{ "%.2f"|format(result.execution_time_seconds) }}s</div>
            </div>
            <div class="status-card">
                <div class="label">Iterations</div>
                <div class="value">{{ result.total_iterations }}</div>
            </div>
            <div class="status-card {{ 'warning' if metrics.failed_tool_calls > 0 else 'success' }}">
                <div class="label">Tool Success</div>
                <div class="value">{{ metrics.successful_tool_calls }}/{{ metrics.total_tool_calls }}</div>
            </div>
        </div>
        
        <div class="main-content">
            {% if result.error %}
            <div class="error-card">
                <h3>❌ Error</h3>
                <div class="content">{{ result.error }}</div>
            </div>
            {% endif %}
            
            <div class="prompt-card">
                <div class="prompt-text">{{ prompt }}</div>
            </div>
            
            {% if result.final_answer %}
            <div class="final-answer">
                <h3>🎉 Final Answer</h3>
                <div class="answer-content">{{ result.final_answer }}</div>
            </div>
            {% endif %}
            
            <div class="section">
                <h2 class="section-title">📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Tools Used</h4>
                        <ul class="metric-list">
                            {% for tool in metrics.tools_used %}
                            <li>{{ tool }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    <div class="metric-card">
                        <h4>Success Rate</h4>
                        <div>{{ metrics.successful_tool_calls }}/{{ metrics.total_tool_calls }} tool calls</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{ (metrics.successful_tool_calls / metrics.total_tool_calls * 100) if metrics.total_tool_calls > 0 else 0 }}%"></div>
                        </div>
                    </div>
                    {% if metrics.error_patterns %}
                    <div class="metric-card">
                        <h4>Error Patterns</h4>
                        <ul class="metric-list">
                            {% for error in metrics.error_patterns[:3] %}
                            <li>{{ error[:50] }}{{ '...' if error|length > 50 else '' }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            {% if result.steps %}
            <div class="section">
                <h2 class="section-title">📝 Execution Steps</h2>
                <div class="steps-container">
                    {% for step in result.steps %}
                    <div class="step" id="step-{{ loop.index }}">
                        <div class="step-header" onclick="toggleStep({{ loop.index }})">
                            <div>
                                <div class="step-title">Step {{ loop.index }}: {{ step.action or 'Thinking' }}</div>
                                <div class="step-meta">
                                    <span>Iteration {{ step.iteration }}</span>
                                    {% if step.action %}
                                    <span>Action: {{ step.action }}</span>
                                    {% endif %}
                                </div>
                            </div>
                            <div class="expand-icon">▼</div>
                        </div>
                        <div class="step-content">
                            {% if step.thought %}
                            <div class="step-section">
                                <h4>💭 Thought</h4>
                                <div class="content">{{ step.thought }}</div>
                            </div>
                            {% endif %}
                            
                            {% if step.action_input %}
                            <div class="step-section">
                                <h4>⚙️ Action Input</h4>
                                <div class="content">{{ step.action_input|tojson(indent=2) }}</div>
                            </div>
                            {% endif %}
                            
                            {% if step.observation %}
                            <div class="step-section">
                                <h4>👁️ Observation</h4>
                                <div class="content">{{ _truncate_observation(step.observation) }}</div>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function toggleStep(stepNumber) {
            const step = document.getElementById(`step-${stepNumber}`);
            step.classList.toggle('expanded');
        }
        
        // Auto-expand first step and any error steps
        document.addEventListener('DOMContentLoaded', function() {
            // Expand first step
            const firstStep = document.getElementById('step-1');
            if (firstStep) {
                firstStep.classList.add('expanded');
            }
            
            // Expand steps with errors
            {% for step in result.steps %}
            {% if step.observation and ('error' in step.observation.lower() or 'failed' in step.observation.lower()) %}
            const errorStep{{ loop.index }} = document.getElementById('step-{{ loop.index }}');
            if (errorStep{{ loop.index }}) {
                errorStep{{ loop.index }}.classList.add('expanded');
            }
            {% endif %}
            {% endfor %}
        });
    </script>
</body>
</html>""")

        html_content = template.render(
            result=result,
            metadata=metadata,
            metrics=metrics,
            prompt=result_data.get("prompt", ""),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metadata.get("timestamp", time.time()))),
            _truncate_observation=self._truncate_observation
        )

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content

    def _format_batch_html(self, batch_data: dict[str, Any], output_file: str = None) -> str:
        """Format batch results as HTML."""
        # For now, convert markdown to basic HTML (TODO: create dedicated batch HTML template)
        markdown_content = self._format_batch_markdown(batch_data)
        html_content = self._markdown_to_html(markdown_content, "Batch Evaluation Results")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content

    def _format_comparison_html(self, comparison_data: dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as modern, interactive HTML."""
        analysis = comparison_data["analysis"]
        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]
        
        template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Server Comparison: {{ baseline }} vs {{ test }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #2563eb;
            --primary-light: #dbeafe;
            --success: #059669;
            --success-light: #d1fae5;
            --error: #dc2626;
            --error-light: #fee2e2;
            --warning: #d97706;
            --warning-light: #fef3c7;
            --neutral-50: #fafafa;
            --neutral-100: #f5f5f5;
            --neutral-200: #e5e5e5;
            --neutral-300: #d4d4d4;
            --neutral-700: #404040;
            --neutral-800: #262626;
            --neutral-900: #171717;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--neutral-800);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, #1e40af 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            font-size: 1.25rem;
            opacity: 0.9;
        }
        
        .winners-bar {
            background: var(--neutral-50);
            padding: 2rem;
            border-bottom: 1px solid var(--neutral-200);
        }
        
        .winners-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .winner-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            text-align: center;
        }
        
        .winner-card .metric {
            font-size: 0.875rem;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .winner-card .winner {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--success);
        }
        
        .metrics-section {
            padding: 2rem;
        }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .server-card {
            border: 1px solid var(--neutral-200);
            border-radius: 12px;
            overflow: hidden;
        }
        
        .server-header {
            padding: 1rem 1.5rem;
            font-weight: 600;
            font-size: 1.25rem;
        }
        
        .server-header.baseline {
            background: var(--primary-light);
            color: var(--primary);
        }
        
        .server-header.test {
            background: var(--warning-light);
            color: var(--warning);
        }
        
        .server-metrics {
            padding: 1.5rem;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--neutral-200);
        }
        
        .metric-row:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 500;
            color: var(--neutral-700);
        }
        
        .metric-value {
            font-weight: 600;
            color: var(--neutral-900);
        }
        
        .results-section {
            padding: 2rem;
            background: var(--neutral-50);
        }
        
        .section-title {
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--neutral-900);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .result-comparison {
            background: white;
            border-radius: 12px;
            margin-bottom: 2rem;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .result-header {
            background: var(--neutral-100);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--neutral-200);
        }
        
        .result-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--neutral-900);
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            min-height: 400px;
        }
        
        .result-panel {
            padding: 1.5rem;
        }
        
        .result-panel:first-child {
            border-right: 1px solid var(--neutral-200);
        }
        
        .panel-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .panel-title {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-success {
            background: var(--success-light);
            color: var(--success);
        }
        
        .status-error {
            background: var(--error-light);
            color: var(--error);
        }
        
        .result-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 0.75rem;
            background: var(--neutral-50);
            border-radius: 8px;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--neutral-700);
            margin-bottom: 0.25rem;
        }
        
        .stat-value {
            font-weight: 600;
            color: var(--neutral-900);
        }
        
        .final-answer {
            background: var(--neutral-50);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .final-answer h4 {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .answer-content {
            white-space: pre-wrap;
            font-size: 0.875rem;
            line-height: 1.5;
        }
        
        .steps-summary {
            background: var(--neutral-50);
            border-radius: 8px;
            padding: 1rem;
        }
        
        .steps-summary h4 {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .step-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--neutral-200);
            font-size: 0.875rem;
        }
        
        .step-item:last-child {
            border-bottom: none;
        }
        
        .error-display {
            background: var(--error-light);
            border: 1px solid var(--error);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .error-display h4 {
            color: var(--error);
            margin-bottom: 0.5rem;
        }
        
        @media (max-width: 1024px) {
            .comparison-grid,
            .result-grid {
                grid-template-columns: 1fr;
            }
            
            .result-panel:first-child {
                border-right: none;
                border-bottom: 1px solid var(--neutral-200);
            }
        }
        
        @media (max-width: 768px) {
            body {
                padding: 1rem 0.5rem;
            }
            
            .header {
                padding: 1.5rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .winners-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Server Comparison</h1>
            <div class="subtitle">{{ baseline }} vs {{ test }}</div>
        </div>
        
        <div class="winners-bar">
            <div class="winners-grid">
                <div class="winner-card">
                    <div class="metric">Success Rate Winner</div>
                    <div class="winner">🏆 {{ analysis.winner.success_rate }}</div>
                </div>
                <div class="winner-card">
                    <div class="metric">Speed Winner</div>
                    <div class="winner">⚡ {{ analysis.winner.speed }}</div>
                </div>
                <div class="winner-card">
                    <div class="metric">Efficiency Winner</div>
                    <div class="winner">🎯 {{ analysis.winner.efficiency }}</div>
                </div>
                <div class="winner-card">
                    <div class="metric">Reliability Winner</div>
                    <div class="winner">🔒 {{ analysis.winner.reliability }}</div>
                </div>
                <div class="winner-card">
                    <div class="metric">Token Efficiency Winner</div>
                    <div class="winner">💎 {{ analysis.winner.token_efficiency }}</div>
                </div>
            </div>
        </div>
        
        <div class="metrics-section">
            <div class="comparison-grid">
                <div class="server-card">
                    <div class="server-header baseline">{{ baseline }}</div>
                    <div class="server-metrics">
                        <div class="metric-row">
                            <span class="metric-label">Success Rate</span>
                            <span class="metric-value">{{ analysis.success_counts[baseline] }}/{{ analysis.total_prompts }} ({{ "%.1f"|format(analysis.success_rates[baseline] * 100) }}%)</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Avg Execution Time</span>
                            <span class="metric-value">{{ "%.2f"|format(analysis.average_execution_times[baseline]) }}s</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Avg Iterations</span>
                            <span class="metric-value">{{ "%.1f"|format(analysis.average_iterations[baseline]) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Failed Tool Calls</span>
                            <span class="metric-value">{{ analysis.total_failed_tool_calls[baseline] or 0 }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Total Tokens</span>
                            <span class="metric-value">{{ "{:,}".format(analysis.total_token_usage[baseline]) if analysis.total_token_usage[baseline] else 'N/A' }}</span>
                        </div>
                    </div>
                </div>
                
                <div class="server-card">
                    <div class="server-header test">{{ test }}</div>
                    <div class="server-metrics">
                        <div class="metric-row">
                            <span class="metric-label">Success Rate</span>
                            <span class="metric-value">{{ analysis.success_counts[test] }}/{{ analysis.total_prompts }} ({{ "%.1f"|format(analysis.success_rates[test] * 100) }}%)</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Avg Execution Time</span>
                            <span class="metric-value">{{ "%.2f"|format(analysis.average_execution_times[test]) }}s</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Avg Iterations</span>
                            <span class="metric-value">{{ "%.1f"|format(analysis.average_iterations[test]) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Failed Tool Calls</span>
                            <span class="metric-value">{{ analysis.total_failed_tool_calls[test] or 0 }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Total Tokens</span>
                            <span class="metric-value">{{ "{:,}".format(analysis.total_token_usage[test]) if analysis.total_token_usage[test] else 'N/A' }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="results-section">
            <h2 class="section-title">📊 Detailed Results Comparison</h2>
            
            {% for b_result, t_result in zip(comparison_data.detailed_results[baseline], comparison_data.detailed_results[test]) %}
            <div class="result-comparison">
                <div class="result-header">
                    <div class="result-title">{{ b_result.prompt_file }}</div>
                </div>
                
                <div class="result-grid">
                    <div class="result-panel">
                        <div class="panel-header">
                            <div class="panel-title">{{ baseline }}</div>
                            <div class="status-badge {{ 'status-success' if b_result.success else 'status-error' }}">
                                {{ '✅ Success' if b_result.success else '❌ Failed' }}
                            </div>
                        </div>
                        
                        <div class="result-stats">
                            <div class="stat-item">
                                <div class="stat-label">Time</div>
                                <div class="stat-value">{{ "%.2f"|format(b_result.execution_time) }}s</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Iterations</div>
                                <div class="stat-value">{{ b_result.iterations }}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Failed Tools</div>
                                <div class="stat-value">{{ b_result.failed_tool_calls or 0 }}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Steps</div>
                                <div class="stat-value">{{ b_result.steps }}</div>
                            </div>
                        </div>
                        
                        {% if b_result.error %}
                        <div class="error-display">
                            <h4>❌ Error</h4>
                            <div>{{ b_result.error }}</div>
                        </div>
                        {% endif %}
                        
                        {% if b_result.final_answer %}
                        <div class="final-answer">
                            <h4>🎉 Final Answer</h4>
                            <div class="answer-content">{{ b_result.final_answer[:500] }}{{ '...' if b_result.final_answer|length > 500 else '' }}</div>
                        </div>
                        {% endif %}
                        
                        <div class="steps-summary">
                            <h4>📝 Execution Steps</h4>
                            {% for step in b_result.detailed_steps[:5] %}
                            <div class="step-item">{{ loop.index }}. {{ step.action or 'Thinking' }}</div>
                            {% endfor %}
                            {% if b_result.detailed_steps|length > 5 %}
                            <div class="step-item">... and {{ b_result.detailed_steps|length - 5 }} more steps</div>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div class="result-panel">
                        <div class="panel-header">
                            <div class="panel-title">{{ test }}</div>
                            <div class="status-badge {{ 'status-success' if t_result.success else 'status-error' }}">
                                {{ '✅ Success' if t_result.success else '❌ Failed' }}
                            </div>
                        </div>
                        
                        <div class="result-stats">
                            <div class="stat-item">
                                <div class="stat-label">Time</div>
                                <div class="stat-value">{{ "%.2f"|format(t_result.execution_time) }}s</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Iterations</div>
                                <div class="stat-value">{{ t_result.iterations }}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Failed Tools</div>
                                <div class="stat-value">{{ t_result.failed_tool_calls or 0 }}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Steps</div>
                                <div class="stat-value">{{ t_result.steps }}</div>
                            </div>
                        </div>
                        
                        {% if t_result.error %}
                        <div class="error-display">
                            <h4>❌ Error</h4>
                            <div>{{ t_result.error }}</div>
                        </div>
                        {% endif %}
                        
                        {% if t_result.final_answer %}
                        <div class="final-answer">
                            <h4>🎉 Final Answer</h4>
                            <div class="answer-content">{{ t_result.final_answer[:500] }}{{ '...' if t_result.final_answer|length > 500 else '' }}</div>
                        </div>
                        {% endif %}
                        
                        <div class="steps-summary">
                            <h4>📝 Execution Steps</h4>
                            {% for step in t_result.detailed_steps[:5] %}
                            <div class="step-item">{{ loop.index }}. {{ step.action or 'Thinking' }}</div>
                            {% endfor %}
                            {% if t_result.detailed_steps|length > 5 %}
                            <div class="step-item">... and {{ t_result.detailed_steps|length - 5 }} more steps</div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>""")

        html_content = template.render(
            comparison_data=comparison_data,
            analysis=analysis,
            baseline=baseline,
            test=test,
            zip=zip,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(comparison_data.get("metadata", {}).get("timestamp", time.time())))
        )

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content

    def _markdown_to_html(self, markdown_content: str, title: str) -> str:
        """Convert markdown to compact, information-dense HTML."""
        import re
        
        # Process markdown with better HTML structure
        lines = markdown_content.split('\n')
        html_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Headers with collapsible sections
            if line.startswith('### '):
                header_text = line[4:]
                header_id = re.sub(r'[^a-zA-Z0-9]', '', header_text.lower())
                html_lines.append(f'<details class="subsection"><summary class="h3">{header_text}</summary><div class="content">')
            elif line.startswith('## '):
                # Close any open details first
                if html_lines and '<details' in html_lines[-1]:
                    html_lines.append('</div></details>')
                header_text = line[3:]
                html_lines.append(f'<h2>{header_text}</h2>')
            elif line.startswith('# '):
                header_text = line[2:]
                html_lines.append(f'<h1>{header_text}</h1>')
            elif line.startswith('- **'):
                # Summary list items
                match = re.match(r'- \*\*(.*?)\*\*: (.*)', line)
                if match:
                    key, value = match.groups()
                    html_lines.append(f'<div class="summary-item"><span class="key">{key}:</span> <span class="value">{value}</span></div>')
                else:
                    html_lines.append(f'<div class="list-item">{line[2:]}</div>')
            elif line.startswith('- '):
                html_lines.append(f'<div class="list-item">{line[2:]}</div>')
            elif line.startswith('```'):
                # Skip code block markers, we'll handle content
                continue
            elif line.startswith('**') and line.endswith('**'):
                content = line[2:-2]
                html_lines.append(f'<div class="section-label">{content}</div>')
            else:
                # Regular content with word wrapping
                html_lines.append(f'<div class="content-line">{line}</div>')
        
        # Close any remaining open details
        if html_lines and any('<details' in line for line in html_lines):
            html_lines.append('</div></details>')
        
        html_content = '\n'.join(html_lines)

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
