"""Results formatting and output utilities."""

import json
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

    def _format_single_html(self, result_data: dict[str, Any], output_file: str = None) -> str:
        """Format single result as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_single_markdown(result_data)
        html_content = self._markdown_to_html(markdown_content, "Single Evaluation Result")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content

    def _format_batch_html(self, batch_data: dict[str, Any], output_file: str = None) -> str:
        """Format batch results as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_batch_markdown(batch_data)
        html_content = self._markdown_to_html(markdown_content, "Batch Evaluation Results")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content

    def _format_comparison_html(self, comparison_data: dict[str, Any], output_file: str = None) -> str:
        """Format comparison results as HTML."""
        # For now, convert markdown to basic HTML
        markdown_content = self._format_comparison_markdown(comparison_data)
        html_content = self._markdown_to_html(markdown_content, "Server Comparison Results")

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
