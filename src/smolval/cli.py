"""Command-line interface for smolval."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import click

from .agent import ClaudeCodeAgent
from .output_manager import OutputManager


def _generate_output_path(prompt_file: Path, output_format: str) -> Path:
    """Generate timestamped output path in results directory."""
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get prompt file name without extension
    prompt_name = prompt_file.stem

    # Determine file extension
    extension_map = {"json": "json", "markdown": "md", "csv": "csv", "html": "html"}
    extension = extension_map.get(output_format, output_format)

    # Generate final path
    filename = f"{timestamp}_{prompt_name}.{extension}"
    return results_dir / filename


def _show_banner() -> None:
    """Show ASCII banner for smolval."""
    banner = """
    ███████╗███╗   ███╗ ██████╗ ██╗    ██╗   ██╗ █████╗ ██╗
    ██╔════╝████╗ ████║██╔═══██╗██║    ██║   ██║██╔══██╗██║
    ███████╗██╔████╔██║██║   ██║██║    ██║   ██║███████║██║
    ╚════██║██║╚██╔╝██║██║   ██║██║    ╚██╗ ██╔╝██╔══██║██║
    ███████║██║ ╚═╝ ██║╚██████╔╝███████╗╚████╔╝ ██║  ██║███████╗
    ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═══╝  ╚═╝  ╚═╝╚══════╝

    Lightweight MCP server evaluation using Claude Code CLI
    """
    click.echo(click.style(banner, fg="blue"))


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging for troubleshooting")
@click.option("--no-banner", is_flag=True, help="Skip the ASCII banner display")
def cli(debug: bool, no_banner: bool) -> None:
    """Smolval - Lightweight MCP server evaluation with Claude Code.

    Smolval is a Python wrapper around Claude Code CLI that simplifies evaluating
    MCP (Model Context Protocol) servers. It executes prompts through Claude Code
    and provides structured output in multiple formats.

    Examples:
      # Evaluate with default settings (auto-saves to results/ directory)
      smolval eval prompts/test.txt

      # Use custom MCP config with JSON output
      smolval eval prompts/test.txt --mcp-config custom.mcp.json --format json

      # Verbose mode with extended timeout
      smolval eval prompts/complex-task.txt --verbose --timeout 600

    For more information, visit: https://github.com/austinlparker/smolval
    """
    if not no_banner:
        _show_banner()

    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@cli.command()
@click.argument(
    "prompt_file", type=click.Path(exists=True, path_type=Path), metavar="PROMPT_FILE"
)
@click.option(
    "--mcp-config",
    type=click.Path(path_type=Path),
    default=".mcp.json",
    show_default=True,
    help="Path to MCP configuration file. If not found, Claude Code will use built-in tools only.",
)
@click.option(
    "--timeout",
    type=int,
    default=900,
    show_default=True,
    help="Maximum time in seconds to wait for Claude Code execution. Use 0 for no timeout.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown", "csv", "html"]),
    default="markdown",
    show_default=True,
    help="Output format: json (structured data), markdown (human-readable), csv (tabular), html (rich web format).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Custom output file path. If not specified, results are saved to results/<timestamp>_<prompt-name>.<format>.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose mode: show detailed progress, Claude CLI output, and execution summary.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable the progress spinner. Useful for scripts or when redirecting output.",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file for environment variables. Defaults to .env in current directory.",
)
@click.option(
    "--allowed-tools",
    type=str,
    help="Comma-separated list of additional MCP tool prefixes to allow (e.g., 'mcp__sqlite,mcp__github'). Built-in tools are always allowed.",
)
def eval(
    prompt_file: Path,
    mcp_config: Path,
    timeout: int,
    output_format: str,
    output: Path | None,
    verbose: bool,
    no_progress: bool,
    env_file: Path | None,
    allowed_tools: str | None,
) -> None:
    """Evaluate a prompt file using Claude Code CLI with optional MCP servers.

    This command executes the content of PROMPT_FILE through Claude Code CLI,
    optionally using MCP (Model Context Protocol) servers for enhanced capabilities
    like database access, API integrations, or specialized tools.

    Results are automatically saved to timestamped files in the results/ directory
    unless a custom output path is specified. The filename format is:
    results/<YYYYMMDD_HHMMSS>_<prompt-name>.<extension>

    PROMPT_FILE should contain the text prompt you want Claude to process.
    The prompt can request file operations, web searches, or MCP server interactions.

    \b
    Examples:
      # Basic evaluation with default settings
      smolval eval prompts/simple-task.txt

      # Use custom MCP config for database access
      smolval eval prompts/database-query.txt --mcp-config configs/postgres.mcp.json

      # Generate structured JSON output (auto-saved to results/ directory)
      smolval eval prompts/analysis.txt --format json

      # Verbose mode for debugging or detailed monitoring
      smolval eval prompts/complex-task.txt --verbose --timeout 1800

      # Custom output location
      smolval eval prompts/batch-task.txt --format csv --output batch-results.csv

      # Allow specific MCP tools beyond built-in tools
      smolval eval prompts/database-task.txt --allowed-tools "mcp__sqlite,mcp__postgres"

    \b
    Requirements:
      - Docker installed on host system
      - ANTHROPIC_API_KEY environment variable must be set
      - MCP servers (optional): Only needed for functionality beyond Claude's built-in tools

    \b
    Output Formats:
      json      - Structured data with execution details, steps, and metadata
      markdown  - Human-readable format with formatted execution summary
      csv       - Tabular format suitable for spreadsheets and data analysis
      html      - Rich web format with interactive timeline and detailed views
    """

    async def run_evaluation() -> None:
        # Read the prompt
        try:
            prompt_content = prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            click.echo(f"Error reading prompt file: {e}", err=True)
            sys.exit(1)

        if verbose:
            click.echo(f"📝 Loaded prompt from: {prompt_file}")
            click.echo(f"🔧 Using MCP config: {mcp_config}")
            click.echo(f"⏱️  Timeout: {timeout}s")
            if env_file:
                click.echo(f"🌍 Environment file: {env_file}")
            if allowed_tools:
                click.echo(f"🔓 Tool permissions: Built-in tools + {allowed_tools}")
            else:
                click.echo("🔓 Tool permissions: Built-in tools only")
            click.echo()

        # Create agent
        agent = ClaudeCodeAgent(
            mcp_config_path=str(mcp_config),
            timeout_seconds=timeout,
            verbose=verbose,
            env_file=str(env_file) if env_file else None,
            allowed_mcp_tools=allowed_tools,
        )

        # Run evaluation
        if verbose:
            click.echo("🚀 Starting evaluation...")

        try:
            # Use progress indicator unless disabled or in verbose mode
            show_progress = not (verbose or no_progress)
            result = await agent.run(prompt_content, show_progress=show_progress)
        except Exception as e:
            click.echo(f"Error during evaluation: {e}", err=True)
            sys.exit(1)

        # Format output
        output_manager = OutputManager()
        formatted_output = output_manager.format_result(result, output_format)

        # Determine output path - use custom path or auto-generate
        if output:
            output_path = output
        else:
            output_path = _generate_output_path(prompt_file, output_format)

        # Write output to file
        try:
            output_path.write_text(formatted_output, encoding="utf-8")
            click.echo(f"💾 Results saved to: {output_path}")
        except Exception as e:
            click.echo(f"Error writing output: {e}", err=True)
            sys.exit(1)

        # Show summary
        if verbose:
            click.echo()
            status = "✅ Success" if result.success else "❌ Failed"
            click.echo(
                f"{status} - {result.total_iterations} steps - {result.execution_time_seconds:.2f}s"
            )

    # Run the async evaluation
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    cli()
