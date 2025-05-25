"""Results formatting and output utilities."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from jinja2 import Template


class ResultsFormatter:
    """Formats evaluation results for different output formats."""
    
    def __init__(self, format_type: str = "json"):
        """Initialize the results formatter."""
        if format_type not in ("json", "csv", "markdown", "html"):
            raise ValueError(f"Unsupported format: {format_type}")
        self.format_type = format_type
    
    def format_single_result(self, result_data: Dict[str, Any], output_file: str = None) -> str:
        """Format a single evaluation result."""
        if self.format_type == "json":
            return self._format_json(result_data, output_file)
        elif self.format_type == "csv":
            return self._format_single_csv(result_data, output_file)
        elif self.format_type == "markdown":
            return self._format_single_markdown(result_data, output_file)
        elif self.format_type == "html":
            return self._format_single_html(result_data, output_file)
    
    def format_batch_results(self, batch_data: Dict[str, Any], output_file: str = None) -> str:
        """Format batch evaluation results."""
        if self.format_type == "json":
            return self._format_json(batch_data, output_file)
        elif self.format_type == "csv":
            return self._format_batch_csv(batch_data, output_file)
        elif self.format_type == "markdown":
            return self._format_batch_markdown(batch_data, output_file)
        elif self.format_type == "html":
            return self._format_batch_html(batch_data, output_file)
    
    def format_comparison_results(self, comparison_data: Dict[str, Any], output_file: str = None) -> str:
        """Format comparison results."""
        if self.format_type == "json":
            return self._format_json(comparison_data, output_file)
        elif self.format_type == "csv":
            return self._format_comparison_csv(comparison_data, output_file)
        elif self.format_type == "markdown":
            return self._format_comparison_markdown(comparison_data, output_file)
        elif self.format_type == "html":
            return self._format_comparison_html(comparison_data, output_file)
    
    def _format_json(self, data: Dict[str, Any], output_file: str = None) -> str:
        """Format data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
        return json_str
    
    def _format_single_csv(self, result_data: Dict[str, Any], output_file: str = None) -> str:
        """Format single result as CSV."""
        result = result_data["result"]
        
        # Create a flat structure for CSV
        csv_data = [{
            "prompt_file": result_data.get("metadata", {}).get("prompt_file", ""),
            "success": result["success"],
            "final_answer": result["final_answer"].replace('\n', ' ').replace('\r', ''),
            "total_iterations": result["total_iterations"],
            "execution_time_seconds": result["execution_time_seconds"],
            "error": result.get("error", ""),
            "num_steps": len(result.get("steps", [])),
            "timestamp": result_data.get("metadata", {}).get("timestamp", time.time())
        }]
        
        return self._write_csv(csv_data, output_file)
    
    def _format_batch_csv(self, batch_data: Dict[str, Any], output_file: str = None) -> str:
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
                "error": result.get("error", ""),
                "num_steps": len(result.get("steps", [])),
                "timestamp": result_item.get("metadata", {}).get("timestamp", time.time())
            })
        
        return self._write_csv(csv_data, output_file)
    
    def _format_comparison_csv(self, comparison_data: Dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as CSV."""
        csv_data = []
        
        baseline = comparison_data["baseline_server"]
        test = comparison_data["test_server"]
        
        baseline_results = comparison_data["detailed_results"][baseline]
        test_results = comparison_data["detailed_results"][test]
        
        for i, (b_result, t_result) in enumerate(zip(baseline_results, test_results)):
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
    
    def _write_csv(self, data: List[Dict[str, Any]], output_file: str = None) -> str:
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
    
    def _format_single_markdown(self, result_data: Dict[str, Any], output_file: str = None) -> str:
        """Format single result as Markdown."""
        result = result_data["result"]
        metadata = result_data.get("metadata", {})
        
        template = Template("""# Evaluation Result

## Summary
- **Success**: {{ '✅' if result.success else '❌' }}
- **Execution Time**: {{ "%.2f"|format(result.execution_time_seconds) }}s
- **Iterations**: {{ result.total_iterations }}
- **Steps**: {{ result.steps|length }}
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
**Observation**: {{ step.observation }}
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
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metadata.get("timestamp", time.time())))
        )
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(markdown_content)
        
        return markdown_content
    
    def _format_batch_markdown(self, batch_data: Dict[str, Any], output_file: str = None) -> str:
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

| Prompt File | Success | Time (s) | Iterations | Error |
|-------------|---------|----------|------------|-------|
{% for result_item in batch_data.results %}
| {{ result_item.prompt_file }} | {{ '✅' if result_item.result.success else '❌' }} | {{ "%.2f"|format(result_item.result.execution_time_seconds) }} | {{ result_item.result.total_iterations }} | {{ result_item.result.error or '' }} |
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
    
    def _format_comparison_markdown(self, comparison_data: Dict[str, Any], output_file: str = None) -> str:
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

## Per-Prompt Comparison

| Prompt | {{ baseline }} Success | {{ baseline }} Time | {{ test }} Success | {{ test }} Time | Winner |
|--------|{{ '-' * baseline|length }}-|{{ '-' * baseline|length }}--|{{ '-' * test|length }}-|{{ '-' * test|length }}--|--------|
{% for b_result, t_result in zip(comparison_data.detailed_results[baseline], comparison_data.detailed_results[test]) %}
| {{ b_result.prompt_file }} | {{ '✅' if b_result.success else '❌' }} | {{ "%.2f"|format(b_result.execution_time) }}s | {{ '✅' if t_result.success else '❌' }} | {{ "%.2f"|format(t_result.execution_time) }}s | {{ baseline if (b_result.success and b_result.execution_time <= t_result.execution_time) or (b_result.success and not t_result.success) else test }} |
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
    
    def _format_single_html(self, result_data: Dict[str, Any], output_file: str = None) -> str:
        """Format single result as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_single_markdown(result_data)
        html_content = self._markdown_to_html(markdown_content, "Single Evaluation Result")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _format_batch_html(self, batch_data: Dict[str, Any], output_file: str = None) -> str:
        """Format batch results as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_batch_markdown(batch_data)
        html_content = self._markdown_to_html(markdown_content, "Batch Evaluation Results")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _format_comparison_html(self, comparison_data: Dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_comparison_markdown(comparison_data)
        html_content = self._markdown_to_html(markdown_content, "Server Comparison Results")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _markdown_to_html(self, markdown_content: str, title: str) -> str:
        """Convert markdown to basic HTML."""
        # Basic markdown to HTML conversion
        html_content = markdown_content.replace('\n', '<br>\n')
        html_content = html_content.replace('# ', '<h1>').replace('<br>\n<h1>', '</h1>\n<h1>')
        html_content = html_content.replace('## ', '<h2>').replace('<br>\n<h2>', '</h2>\n<h2>')
        html_content = html_content.replace('### ', '<h3>').replace('<br>\n<h3>', '</h3>\n<h3>')
        html_content = html_content.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        html_content = html_content.replace('```', '<pre>').replace('```', '</pre>')
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        h3 {{ color: #999; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""