"""Results formatting and output utilities."""

import json
from pathlib import Path
from typing import Any

import markdown_it
from jinja2 import Environment, FileSystemLoader


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

    def format_result(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format an evaluation result."""
        if self.format_type == "json":
            return self._format_json(result_data, output_file)
        elif self.format_type == "csv":
            return self._format_csv(result_data, output_file)
        elif self.format_type == "markdown":
            return self._format_markdown(result_data, output_file)
        elif self.format_type == "html":
            return self._format_html(result_data, output_file)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")

    def _format_json(self, data: dict[str, Any], output_file: str | None = None) -> str:
        """Format data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        if output_file:
            with open(output_file, "w") as f:
                f.write(json_str)
        return json_str

    def _format_csv(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format result as CSV."""
        import csv
        import io

        # Extract the result
        if "result" in result_data:
            result = result_data["result"]
        else:
            result = result_data

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "success",
                "execution_time_seconds",
                "total_iterations",
                "failed_tool_calls",
                "successful_tool_calls",
                "final_answer_length",
                "error_message",
                "session_id",
                "model_used",
                "total_cost_usd",
                "mcp_servers_count",
                "mcp_servers_list",
                "tools_count",
                "tools_list",
            ]
        )

        # Write data
        metadata = result.get("metadata", {})
        mcp_servers = metadata.get("mcp_servers_used", [])
        tools = metadata.get("tools_available", [])

        writer.writerow(
            [
                result.get("success", False),
                result.get("execution_time_seconds", 0.0),
                result.get("total_iterations", 0),
                result.get("failed_tool_calls", 0),
                result.get("successful_tool_calls", 0),
                len(result.get("final_answer", "")),
                result.get("error_message", ""),
                metadata.get("session_id", ""),
                metadata.get("model_used", ""),
                metadata.get("total_cost_usd", ""),
                len(mcp_servers),
                ";".join(mcp_servers) if mcp_servers else "",
                len(tools),
                ";".join(tools) if tools else "",
            ]
        )

        csv_str = output.getvalue()
        if output_file:
            with open(output_file, "w") as f:
                f.write(csv_str)
        return csv_str

    def _format_markdown(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format result as Markdown."""
        # Extract the result
        if "result" in result_data:
            result = result_data["result"]
        else:
            result = result_data

        metadata = result.get("metadata", {})

        markdown_lines = [
            "# Evaluation Result",
            "",
            f"**Status:** {'✅ Success' if result.get('success', False) else '❌ Failed'}",
            f"**Execution Time:** {result.get('execution_time_seconds', 0.0):.2f}s",
            f"**Iterations:** {result.get('total_iterations', 0)}",
            f"**Model:** {metadata.get('model_used', 'unknown')}",
            f"**Session ID:** {metadata.get('session_id', 'unknown')}",
            "",
        ]

        if metadata.get("total_cost_usd"):
            markdown_lines.append(f"**Total Cost:** ${metadata['total_cost_usd']:.4f}")
            markdown_lines.append("")

        # Add environment information section
        env_info = []
        if metadata.get("mcp_servers_used"):
            env_info.append(
                f"**MCP Servers:** {', '.join(metadata['mcp_servers_used'])}"
            )
        if metadata.get("tools_available"):
            env_info.append(
                f"**Available Tools:** {', '.join(metadata['tools_available'])}"
            )

        if env_info:
            markdown_lines.extend(["## Environment", ""])
            markdown_lines.extend(env_info)
            markdown_lines.append("")

        if result.get("error_message"):
            markdown_lines.extend(
                ["## Error", "", "```", result["error_message"], "```", ""]
            )

        if result.get("final_answer"):
            markdown_lines.extend(["## Final Answer", "", result["final_answer"], ""])

        if result.get("steps"):
            markdown_lines.extend(["## Execution Steps", ""])

            for i, step in enumerate(result["steps"], 1):
                step_type = step.get("step_type", "unknown")
                content = step.get("content", "N/A")

                markdown_lines.extend(
                    [
                        f"### Step {i} - {step_type.replace('_', ' ').title()}",
                        "",
                        f"**Content:** {content}",
                        "",
                    ]
                )

                if step.get("tool_name"):
                    markdown_lines.append(f"**Tool:** {step['tool_name']}")

                if step.get("tool_output"):
                    markdown_lines.extend(
                        ["", "**Tool Output:**", "```", step["tool_output"], "```", ""]
                    )

        markdown_str = "\n".join(markdown_lines)
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_str)
        return markdown_str

    def _format_html(
        self, result_data: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format result as HTML."""
        # Extract the result
        if "result" in result_data:
            result = result_data["result"]
        else:
            result = result_data

        try:
            template = self.jinja_env.get_template("single_result.html")
        except Exception:
            # Fallback to simple HTML if template not found
            return self._format_simple_html(result, output_file)

        # Prepare data for template
        metadata = result.get("metadata", {})
        template_data = {
            "result": result,
            "success": result.get("success", False),
            "execution_time": result.get("execution_time_seconds", 0.0),
            "iterations": result.get("total_iterations", 0),
            "failed_tool_calls": result.get("failed_tool_calls", 0),
            "successful_tool_calls": result.get("successful_tool_calls", 0),
            "final_answer": result.get("final_answer", ""),
            "error_message": result.get("error_message", ""),
            "steps": result.get("steps", []),
            "metadata": {
                "prompt_file": "Evaluation Result",
                "timestamp": metadata.get("execution_start", "Unknown"),
                "session_id": metadata.get("session_id", "Unknown"),
                "model_used": metadata.get("model_used", "Unknown"),
                "total_cost_usd": metadata.get("total_cost_usd"),
                "mcp_servers_used": metadata.get("mcp_servers_used", []),
                "tools_available": metadata.get("tools_available", []),
            },
        }

        html_str = template.render(**template_data)
        if output_file:
            with open(output_file, "w") as f:
                f.write(html_str)
        return html_str

    def _format_simple_html(
        self, result: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Format result as simple HTML when template is not available."""
        status = "Success" if result.get("success", False) else "Failed"
        status_class = "success" if result.get("success", False) else "error"

        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Evaluation Result</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 40px; }",
            "        .success { color: green; }",
            "        .error { color: red; }",
            "        .step { margin: 20px 0; padding: 15px; border-left: 3px solid #ddd; }",
            "        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Evaluation Result</h1>",
            f"    <p><strong>Status:</strong> <span class='{status_class}'>{status}</span></p>",
            f"    <p><strong>Execution Time:</strong> {result.get('execution_time_seconds', 0.0):.2f}s</p>",
            f"    <p><strong>Iterations:</strong> {result.get('total_iterations', 0)}</p>",
            f"    <p><strong>Model:</strong> {result.get('metadata', {}).get('model_used', 'unknown')}</p>",
        ]

        if result.get("error_message"):
            html_lines.extend(
                ["    <h2>Error</h2>", f"    <pre>{result['error_message']}</pre>"]
            )

        if result.get("final_answer"):
            html_lines.extend(
                [
                    "    <h2>Final Answer</h2>",
                    f"    <div>{result['final_answer']}</div>",
                ]
            )

        if result.get("steps"):
            html_lines.append("    <h2>Execution Steps</h2>")
            for i, step in enumerate(result["steps"], 1):
                html_lines.extend(
                    [
                        "    <div class='step'>",
                        f"        <h3>Step {i} - {step.get('step_type', 'unknown').replace('_', ' ').title()}</h3>",
                        f"        <p><strong>Content:</strong> {step.get('content', 'N/A')}</p>",
                    ]
                )

                if step.get("tool_name"):
                    html_lines.append(
                        f"        <p><strong>Tool:</strong> {step['tool_name']}</p>"
                    )

                if step.get("tool_output"):
                    html_lines.extend(
                        [
                            "        <p><strong>Tool Output:</strong></p>",
                            f"        <pre>{step['tool_output']}</pre>",
                        ]
                    )

                html_lines.append("    </div>")

        html_lines.extend(["</body>", "</html>"])

        html_str = "\n".join(html_lines)
        if output_file:
            with open(output_file, "w") as f:
                f.write(html_str)
        return html_str
