"""Command-line interface for smolval."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import click

from .agent import Agent
from .config import Config
from .llm_client import LLMClient
from .mcp_client import MCPClientManager
from .output_manager import OutputManager


async def _connect_mcp_servers_silently(
    mcp_manager: MCPClientManager, server_configs: list, debug: bool = False
) -> None:
    """Connect to MCP servers while suppressing OS-level stderr noise."""
    if debug:
        # Don't suppress stderr in debug mode
        for server_config in server_configs:
            await mcp_manager.connect(server_config, debug=True)
    else:
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            for server_config in server_configs:
                await mcp_manager.connect(server_config, debug=False)
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
            os.close(devnull)


def _show_banner() -> None:
    """Show ASCII banner for smolval."""
    banner = """
    ███████╗███╗   ███╗ ██████╗ ██╗    ██╗   ██╗ █████╗ ██╗
    ██╔════╝████╗ ████║██╔═══██╗██║    ██║   ██║██╔══██╗██║
    ███████╗██╔████╔██║██║   ██║██║    ██║   ██║███████║██║
    ╚════██║██║╚██╔╝██║██║   ██║██║    ╚██╗ ██╔╝██╔══██║██║
    ███████║██║ ╚═╝ ██║╚██████╔╝███████╗╚████╔╝ ██║  ██║███████╗
    ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═══╝  ╚═╝  ╚═╝╚══════╝

    🤖 A lightweight MCP server evaluation agent
    """
    click.echo(click.style(banner, fg="cyan", bold=True))


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--no-banner", is_flag=True, help="Disable ASCII banner")
@click.pass_context
def main(ctx: click.Context, verbose: bool, debug: bool, no_banner: bool) -> None:
    """smolval: A lightweight MCP server evaluation agent."""
    # Store flags in context for commands to use
    ctx.ensure_object(dict)
    ctx.obj["no_banner"] = no_banner
    ctx.obj["debug"] = debug

    # Configure logging
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@main.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default="config/example-anthropic.yaml",
    help="Configuration file path",
)
@click.option("--output-dir", help="Output directory for results")
@click.option("--run-name", help="Name for this evaluation run")
@click.pass_context
def eval(
    ctx: click.Context, prompt_file: str, config: str, output_dir: str, run_name: str
) -> None:
    """Evaluate MCP servers using a prompt file."""
    # Show banner unless disabled
    if not ctx.obj.get("no_banner", False):
        _show_banner()

    debug = ctx.obj.get("debug", False)
    asyncio.run(_run_eval(prompt_file, config, output_dir, run_name, debug))


async def _run_eval(
    prompt_file: str,
    config_path: str,
    output_dir: str,
    run_name: str,
    debug: bool = False,
) -> None:
    """Run evaluation asynchronously."""
    logger = logging.getLogger(__name__)
    mcp_manager = None

    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        logger.debug("Loading configuration from %s", config_path)
        config = Config.from_yaml(Path(config_path))

        # Load prompt
        click.echo(f"Loading prompt from: {prompt_file}")
        with open(prompt_file) as f:
            prompt = f.read().strip()

        # Initialize components
        click.echo("Initializing LLM client...")
        llm_client = LLMClient(config.llm)

        click.echo("Initializing MCP client manager...")
        mcp_manager = MCPClientManager()

        # Connect to MCP servers (suppress server startup noise)
        click.echo("Connecting to MCP servers...")
        await _connect_mcp_servers_silently(mcp_manager, config.mcp_servers, debug)

        # Initialize agent
        click.echo("Initializing agent...")
        agent = Agent(config, llm_client, mcp_manager)

        # Run evaluation with progress indicator
        click.echo("Running evaluation...")

        # Progress indicator
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spinner_idx = 0
        current_step = 0

        def show_progress(step: int, max_step: int) -> None:
            nonlocal spinner_idx, current_step
            current_step = step
            char = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1

            # Clear line and show progress
            sys.stdout.write(f"\r{char} Step {step}/{max_step}")
            sys.stdout.flush()

        start_time = time.time()
        result = await agent.run(prompt, progress_callback=show_progress)
        end_time = time.time()

        # Clear progress line
        sys.stdout.write(f"\r✅ Completed {current_step} steps\n")
        sys.stdout.flush()

        # Format result
        result_data = {
            "prompt": prompt,
            "result": {
                "success": result.success,
                "final_answer": result.final_answer,
                "steps": [step.model_dump() for step in result.steps],
                "total_iterations": result.total_iterations,
                "error": result.error,
                "execution_time_seconds": result.execution_time_seconds,
                "token_usage": result.token_usage,
                "failed_tool_calls": result.failed_tool_calls,
            },
            "llm_responses": result.llm_responses,
            "metadata": {
                "config_file": config_path,
                "prompt_file": prompt_file,
                "timestamp": time.time(),
                "duration_seconds": end_time - start_time,
                "failed_tool_calls": result.failed_tool_calls,
            },
        }

        # Output result using OutputManager
        if output_dir:
            # Use output manager with specified directory
            output_manager = OutputManager(output_dir)
        else:
            # Use default directory and show console output too
            output_manager = OutputManager("results")

        # Create run directory
        if run_name:
            output_manager.create_run_directory(run_name)
        else:
            output_manager.create_run_directory()

        # Generate eval name from prompt file
        prompt_path = Path(prompt_file)
        eval_name = prompt_path.stem

        # Write results in all formats
        output_files = output_manager.write_evaluation_results(result_data, eval_name)

        click.echo(f"Results written to: {output_manager.get_run_directory()}")
        for format_type, file_path in output_files.items():
            click.echo(f"  {format_type.upper()}: {file_path}")

        if result.success:
            click.echo("✅ Evaluation completed successfully")
        else:
            click.echo("❌ Evaluation failed")
            if result.error:
                click.echo(f"Error: {result.error}")

    except Exception as e:
        click.echo(f"❌ Evaluation failed with error: {e}")
        raise click.ClickException(str(e)) from e

    finally:
        # Always clean up MCP connections
        if mcp_manager is not None:
            try:
                logger.debug("Starting MCP cleanup...")
                # Add timeout to prevent hanging
                await asyncio.wait_for(mcp_manager.close(), timeout=5.0)
                logger.debug("MCP cleanup completed")
            except TimeoutError:
                logger.warning("MCP cleanup timed out - forcing exit")
            except Exception as cleanup_error:
                logger.warning("Error during MCP cleanup: %s", cleanup_error)


@main.command()
@click.argument("prompts_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default="config/example-anthropic.yaml",
    help="Configuration file path",
)
@click.option("--output-dir", help="Output directory for results")
@click.option("--run-name", help="Name for this batch run")
@click.option("--servers", help="Comma-separated list of server names to filter")
@click.pass_context
def batch(
    ctx: click.Context,
    prompts_dir: str,
    config: str,
    output_dir: str,
    run_name: str,
    servers: str,
) -> None:
    """Run batch evaluation against multiple prompts."""
    # Show banner unless disabled
    if not ctx.obj.get("no_banner", False):
        _show_banner()

    debug = ctx.obj.get("debug", False)
    asyncio.run(_run_batch(prompts_dir, config, output_dir, run_name, servers, debug))


async def _run_batch(
    prompts_dir: str,
    config_path: str,
    output_dir: str,
    run_name: str,
    servers_filter: str,
    debug: bool = False,
) -> None:
    """Run batch evaluation asynchronously."""
    mcp_manager = None

    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        config = Config.from_yaml(Path(config_path))

        # Filter servers if specified
        if servers_filter:
            server_names = [s.strip() for s in servers_filter.split(",")]
            config.mcp_servers = [
                s for s in config.mcp_servers if s.name in server_names
            ]
            click.echo(f"Filtering to servers: {server_names}")

        # Find all prompt files
        prompts_path = Path(prompts_dir)
        prompt_files = list(prompts_path.glob("*.txt")) + list(
            prompts_path.glob("*.md")
        )

        if not prompt_files:
            raise click.ClickException(f"No prompt files found in {prompts_dir}")

        click.echo(f"Found {len(prompt_files)} prompt files")

        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Initialize components once
        click.echo("Initializing LLM client...")
        llm_client = LLMClient(config.llm)

        click.echo("Initializing MCP client manager...")
        mcp_manager = MCPClientManager()

        # Connect to MCP servers (suppress server startup noise)
        click.echo("Connecting to MCP servers...")
        await _connect_mcp_servers_silently(mcp_manager, config.mcp_servers, debug)

        # Initialize agent
        agent = Agent(config, llm_client, mcp_manager)

        # Run evaluations
        results = []
        for i, prompt_file in enumerate(prompt_files, 1):
            click.echo(f"\n[{i}/{len(prompt_files)}] Evaluating: {prompt_file.name}")

            try:
                # Load prompt
                with open(prompt_file) as f:
                    prompt = f.read().strip()

                # Run evaluation
                start_time = time.time()
                result = await agent.run(prompt)
                end_time = time.time()

                # Format result
                result_data = {
                    "prompt_file": str(prompt_file.name),
                    "prompt": prompt,
                    "result": {
                        "success": result.success,
                        "final_answer": result.final_answer,
                        "steps": [step.model_dump() for step in result.steps],
                        "total_iterations": result.total_iterations,
                        "error": result.error,
                        "execution_time_seconds": result.execution_time_seconds,
                        "token_usage": result.token_usage,
                    },
                    "metadata": {
                        "timestamp": time.time(),
                        "duration_seconds": end_time - start_time,
                    },
                }

                results.append(result_data)

                # Write individual result if output directory specified
                if output_dir:
                    output_file = output_path / f"{prompt_file.stem}_result.json"
                    with open(output_file, "w") as f:
                        json.dump(result_data, f, indent=2)

                status = "✅" if result.success else "❌"
                click.echo(f"   {status} {result.execution_time_seconds:.2f}s")

            except Exception as e:
                click.echo(f"   ❌ Error: {e}")
                results.append(
                    {
                        "prompt_file": str(prompt_file.name),
                        "prompt": "",
                        "result": {
                            "success": False,
                            "final_answer": "",
                            "steps": [],
                            "total_iterations": 0,
                            "error": str(e),
                            "execution_time_seconds": 0,
                        },
                        "metadata": {"timestamp": time.time(), "duration_seconds": 0},
                    }
                )

        # Write summary results
        summary = {
            "config_file": config_path,
            "prompts_directory": prompts_dir,
            "total_prompts": len(prompt_files),
            "successful": sum(1 for r in results if bool(r["result"]["success"])),  # type: ignore[misc,index]
            "failed": sum(1 for r in results if not bool(r["result"]["success"])),  # type: ignore[misc,index]
            "results": results,
            "metadata": {
                "timestamp": time.time(),
                "servers": [s.name for s in config.mcp_servers],
            },
        }

        # Always generate output
        base_dir = output_dir or "results"
        output_manager = OutputManager(base_dir)

        # Create run directory
        batch_run_name = run_name or f"batch_{Path(prompts_dir).name}"
        output_manager.create_run_directory(batch_run_name)

        # Write batch results in all formats
        output_files = output_manager.write_batch_results(summary, batch_run_name)

        click.echo(f"\nBatch results written to: {output_manager.get_run_directory()}")
        for format_type, file_path in output_files.items():
            click.echo(f"  {format_type.upper()}: {file_path}")

        # Print summary
        click.echo(f"\n{'='*60}")
        click.echo("BATCH EVALUATION SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total prompts: {summary['total_prompts']}")
        click.echo(f"Successful: {summary['successful']} ✅")
        click.echo(f"Failed: {summary['failed']} ❌")
        click.echo(
            f"Success rate: {summary['successful']/summary['total_prompts']*100:.1f}%"  # type: ignore[operator]
        )

    except Exception as e:
        click.echo(f"❌ Batch evaluation failed: {e}")
        raise click.ClickException(str(e)) from e

    finally:
        # Always clean up MCP connections
        if mcp_manager is not None:
            try:
                logger = logging.getLogger(__name__)
                logger.debug("Starting batch MCP cleanup...")
                await asyncio.wait_for(mcp_manager.close(), timeout=5.0)
                logger.debug("Batch MCP cleanup completed")
            except TimeoutError:
                logger = logging.getLogger(__name__)
                logger.warning("Batch MCP cleanup timed out - forcing exit")
            except Exception as cleanup_error:
                logger = logging.getLogger(__name__)
                logger.warning("Error during batch MCP cleanup: %s", cleanup_error)


@main.command()
@click.option("--baseline", required=True, help="Baseline server name")
@click.option("--test", required=True, help="Test server name")
@click.argument("prompts_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default="config/example-anthropic.yaml",
    help="Configuration file path",
)
@click.option("--output-dir", help="Output directory for comparison results")
@click.option("--run-name", help="Name for this comparison run")
@click.pass_context
def compare(
    ctx: click.Context,
    baseline: str,
    test: str,
    prompts_dir: str,
    config: str,
    output_dir: str,
    run_name: str,
) -> None:
    """Compare performance between two MCP servers."""
    # Show banner unless disabled
    if not ctx.obj.get("no_banner", False):
        _show_banner()

    asyncio.run(_run_compare(baseline, test, prompts_dir, config, output_dir, run_name))


@main.command()
@click.option("--baseline-config", required=True, help="Baseline provider config file")
@click.option("--test-config", required=True, help="Test provider config file")
@click.argument("prompts_dir", type=click.Path(exists=True))
@click.option("--output-dir", help="Output directory for comparison results")
@click.option("--run-name", help="Name for this comparison run")
@click.pass_context
def compare_providers(
    ctx: click.Context,
    baseline_config: str,
    test_config: str,
    prompts_dir: str,
    output_dir: str,
    run_name: str,
) -> None:
    """Compare performance between different LLM providers."""
    # Show banner unless disabled
    if not ctx.obj.get("no_banner", False):
        _show_banner()

    asyncio.run(
        _run_provider_compare(
            baseline_config, test_config, prompts_dir, output_dir, run_name
        )
    )


async def _run_provider_compare(
    baseline_config_path: str,
    test_config_path: str,
    prompts_dir: str,
    output_dir: str,
    run_name: str,
) -> None:
    """Run provider comparison asynchronously."""
    try:
        # Load configurations
        click.echo(f"Loading baseline config from: {baseline_config_path}")
        baseline_config = Config.from_yaml(Path(baseline_config_path))

        click.echo(f"Loading test config from: {test_config_path}")
        test_config = Config.from_yaml(Path(test_config_path))

        # Find all prompt files
        prompts_path = Path(prompts_dir)
        prompt_files = list(prompts_path.glob("*.txt")) + list(
            prompts_path.glob("*.md")
        )

        if not prompt_files:
            raise click.ClickException(f"No prompt files found in {prompts_dir}")

        click.echo(f"Found {len(prompt_files)} prompt files")

        baseline_provider = (
            f"{baseline_config.llm.provider}:{baseline_config.llm.model}"
        )
        test_provider = f"{test_config.llm.provider}:{test_config.llm.model}"
        click.echo(f"Comparing: {baseline_provider} vs {test_provider}")

        # Run evaluations for both providers
        comparison_results = []

        for config_name, config in [
            ("baseline", baseline_config),
            ("test", test_config),
        ]:
            provider_name = f"{config.llm.provider}:{config.llm.model}"
            click.echo(f"\n--- Running evaluations for {provider_name} ---")
            mcp_manager = None

            try:
                # Initialize components
                llm_client = LLMClient(config.llm)
                mcp_manager = MCPClientManager()

                # Connect to all MCP servers for this config
                for srv_cfg in config.mcp_servers:
                    await mcp_manager.connect(srv_cfg)

                agent = Agent(config, llm_client, mcp_manager)

                # Run evaluations
                provider_results = []
                for i, prompt_file in enumerate(prompt_files, 1):
                    click.echo(f"  [{i}/{len(prompt_files)}] {prompt_file.name}")

                    try:
                        # Load prompt
                        with open(prompt_file) as f:
                            prompt = f.read().strip()

                        # Run evaluation
                        result = await agent.run(prompt)

                        provider_results.append(
                            {
                                "prompt_file": str(prompt_file.name),
                                "prompt": prompt,
                                "success": result.success,
                                "final_answer": result.final_answer,
                                "iterations": result.total_iterations,
                                "execution_time": result.execution_time_seconds,
                                "error": result.error,
                                "steps": len(result.steps),
                                "failed_tool_calls": result.failed_tool_calls,
                                "token_usage": result.token_usage,
                                "detailed_steps": [
                                    step.model_dump() for step in result.steps
                                ],
                                "llm_responses": result.llm_responses,
                            }
                        )

                        status = "✅" if result.success else "❌"
                        click.echo(f"    {status} {result.execution_time_seconds:.2f}s")

                    except Exception as e:
                        click.echo(f"    ❌ Error: {e}")
                        provider_results.append(
                            {
                                "prompt_file": str(prompt_file.name),
                                "prompt": prompt,
                                "success": False,
                                "final_answer": "",
                                "iterations": 0,
                                "execution_time": 0.0,
                                "error": str(e),
                                "steps": 0,
                                "failed_tool_calls": 0,
                                "token_usage": None,
                                "detailed_steps": [],
                                "llm_responses": [],
                            }
                        )

                comparison_results.append(
                    {
                        "provider": provider_name,
                        "config_name": config_name,
                        "results": provider_results,
                    }
                )

            except Exception as e:
                click.echo(f"❌ Provider {provider_name} failed: {e}")
                # Add empty results for failed provider
                comparison_results.append(
                    {
                        "provider": provider_name,
                        "config_name": config_name,
                        "results": [],
                    }
                )

            finally:
                # Clean up MCP connections
                if mcp_manager is not None:
                    try:
                        await asyncio.wait_for(mcp_manager.close(), timeout=5.0)
                    except Exception as cleanup_error:
                        click.echo(f"Warning: MCP cleanup error: {cleanup_error}")

        # Analyze results - handle cases where one or both providers failed
        if len(comparison_results) >= 1:
            # Ensure we have baseline and test results, even if empty
            if len(comparison_results) == 1:
                # Only one provider completed, create empty results for the other
                completed_result = comparison_results[0]
                if completed_result["config_name"] == "baseline":
                    baseline_results: list = completed_result["results"]  # type: ignore[assignment]
                    test_results: list = []
                    baseline_provider_name: str = completed_result["provider"]  # type: ignore[assignment]
                    test_provider_name: str = (
                        f"{test_config.llm.provider}:{test_config.llm.model}"
                    )
                else:
                    baseline_results = []
                    test_results = completed_result["results"]  # type: ignore[assignment]
                    baseline_provider_name = (
                        f"{baseline_config.llm.provider}:{baseline_config.llm.model}"
                    )
                    test_provider_name = completed_result["provider"]  # type: ignore[assignment]
            else:
                # Both providers attempted (may have failed)
                baseline_results = comparison_results[0]["results"]  # type: ignore[index,assignment]
                test_results = comparison_results[1]["results"]  # type: ignore[index,assignment]
                baseline_provider_name = comparison_results[0]["provider"]  # type: ignore[index,assignment]
                test_provider_name = comparison_results[1]["provider"]  # type: ignore[index,assignment]

            # Analyze comparison (handles empty results gracefully)
            analysis = _analyze_comparison(
                baseline_provider_name, test_provider_name, baseline_results, test_results  # type: ignore[arg-type]
            )

            # Generate output
            base_dir = output_dir or "results"
            output_manager = OutputManager(base_dir)

            # Create run directory
            comparison_run_name = (
                run_name
                or f"provider_comparison_{baseline_config.llm.provider}_vs_{test_config.llm.provider}"
            )
            output_manager.create_run_directory(comparison_run_name)

            # Prepare comparison data with failure information
            comparison_data = {
                "baseline_server": baseline_provider_name,  # Match results.py expectations
                "test_server": test_provider_name,  # Match results.py expectations
                "baseline_config": baseline_config_path,
                "test_config": test_config_path,
                "analysis": analysis,
                "detailed_results": {
                    baseline_provider_name: baseline_results,
                    test_provider_name: test_results,
                },
                "provider_failures": {
                    "baseline_failed": len(baseline_results) == 0
                    and len(comparison_results) >= 2,
                    "test_failed": len(test_results) == 0
                    and len(comparison_results) >= 2,
                },
                "timestamp": time.time(),
            }

            # Write comparison results
            output_files = output_manager.write_comparison_results(
                comparison_data, comparison_run_name
            )

            click.echo(
                f"\nComparison results written to: {output_manager.get_run_directory()}"
            )
            for format_type, file_path in output_files.items():
                click.echo(f"  {format_type.upper()}: {file_path}")

            # Print summary with failure information
            if len(baseline_results) == 0:
                click.echo(
                    f"\n⚠️  Baseline provider ({baseline_provider_name}) failed completely"
                )
            if len(test_results) == 0:
                click.echo(
                    f"\n⚠️  Test provider ({test_provider_name}) failed completely"
                )

            _print_comparison_summary(analysis)

        else:
            click.echo(
                "❌ Could not complete comparison - no results from either provider"
            )

    except Exception as e:
        click.echo(f"❌ Provider comparison failed: {e}")
        raise click.ClickException(str(e)) from e


async def _run_compare(
    baseline: str,
    test: str,
    prompts_dir: str,
    config_path: str,
    output_dir: str,
    run_name: str,
) -> None:
    """Run comparison asynchronously."""
    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        config = Config.from_yaml(Path(config_path))

        # Validate servers exist in config
        server_names = [s.name for s in config.mcp_servers]
        if baseline not in server_names:
            raise click.ClickException(
                f"Baseline server '{baseline}' not found in config"
            )
        if test not in server_names:
            raise click.ClickException(f"Test server '{test}' not found in config")

        # Find all prompt files
        prompts_path = Path(prompts_dir)
        prompt_files = list(prompts_path.glob("*.txt")) + list(
            prompts_path.glob("*.md")
        )

        if not prompt_files:
            raise click.ClickException(f"No prompt files found in {prompts_dir}")

        click.echo(f"Found {len(prompt_files)} prompt files")
        click.echo(f"Comparing: {baseline} vs {test}")

        # Run evaluations for both servers
        comparison_results = []

        for server_name in [baseline, test]:
            click.echo(f"\n--- Running evaluations for {server_name} ---")
            mcp_manager = None

            try:
                # Filter config to only this server
                server_config = Config(
                    mcp_servers=[
                        s for s in config.mcp_servers if s.name == server_name
                    ],
                    llm=config.llm,
                    evaluation=config.evaluation,
                )

                # Initialize components
                llm_client = LLMClient(server_config.llm)
                mcp_manager = MCPClientManager()

                # Connect to MCP server
                for srv_cfg in server_config.mcp_servers:
                    await mcp_manager.connect(srv_cfg)

                agent = Agent(server_config, llm_client, mcp_manager)

                # Run evaluations
                server_results = []
                for i, prompt_file in enumerate(prompt_files, 1):
                    click.echo(f"  [{i}/{len(prompt_files)}] {prompt_file.name}")

                    try:
                        # Load prompt
                        with open(prompt_file) as f:
                            prompt = f.read().strip()

                        # Run evaluation
                        result = await agent.run(prompt)

                        server_results.append(
                            {
                                "prompt_file": str(prompt_file.name),
                                "prompt": prompt,
                                "success": result.success,
                                "final_answer": result.final_answer,
                                "iterations": result.total_iterations,
                                "execution_time": result.execution_time_seconds,
                                "error": result.error,
                                "steps": len(result.steps),
                                "failed_tool_calls": result.failed_tool_calls,
                                "token_usage": result.token_usage,
                                "detailed_steps": [
                                    step.model_dump() for step in result.steps
                                ],
                                "llm_responses": result.llm_responses,
                            }
                        )

                        status = "✅" if result.success else "❌"
                        click.echo(f"    {status} {result.execution_time_seconds:.2f}s")

                    except Exception as e:
                        click.echo(f"    ❌ Error: {e}")
                        server_results.append(
                            {
                                "prompt_file": str(prompt_file.name),
                                "prompt": "",
                                "success": False,
                                "final_answer": "",
                                "iterations": 0,
                                "execution_time": 0,
                                "error": str(e),
                                "steps": 0,
                                "failed_tool_calls": 0,
                                "token_usage": None,
                                "detailed_steps": [],
                                "llm_responses": [],
                            }
                        )

                comparison_results.append(
                    {"server_name": server_name, "results": server_results}
                )

            finally:
                # Always clean up MCP connections for this server
                if mcp_manager is not None:
                    try:
                        logger = logging.getLogger(__name__)
                        logger.debug(
                            "Starting compare MCP cleanup for %s...", server_name
                        )
                        await asyncio.wait_for(mcp_manager.close(), timeout=5.0)
                        logger.debug(
                            "Compare MCP cleanup completed for %s", server_name
                        )
                    except TimeoutError:
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            "Compare MCP cleanup timed out for %s - forcing exit",
                            server_name,
                        )
                    except Exception as cleanup_error:
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            "Error during MCP cleanup for %s: %s",
                            server_name,
                            cleanup_error,
                        )

        # Generate comparison analysis
        baseline_results = comparison_results[0]["results"]
        test_results = comparison_results[1]["results"]

        analysis = _analyze_comparison(baseline, test, baseline_results, test_results)  # type: ignore[arg-type]

        # Create final comparison data
        comparison_data = {
            "baseline_server": baseline,
            "test_server": test,
            "prompts_directory": prompts_dir,
            "config_file": config_path,
            "analysis": analysis,
            "detailed_results": {baseline: baseline_results, test: test_results},
            "metadata": {"timestamp": time.time(), "total_prompts": len(prompt_files)},
        }

        # Output results using OutputManager
        base_dir = output_dir or "results"
        output_manager = OutputManager(base_dir)

        # Create run directory
        comparison_run_name = run_name or f"compare_{baseline}_vs_{test}"
        output_manager.create_run_directory(comparison_run_name)

        # Write comparison results in all formats
        output_files = output_manager.write_comparison_results(
            comparison_data, comparison_run_name
        )

        click.echo(
            f"\nComparison results written to: {output_manager.get_run_directory()}"
        )
        for format_type, file_path in output_files.items():
            click.echo(f"  {format_type.upper()}: {file_path}")

        # Print comparison summary
        _print_comparison_summary(analysis)

    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}")
        raise click.ClickException(str(e)) from e


def _analyze_comparison(
    baseline: str, test: str, baseline_results: list, test_results: list
) -> dict:
    """Analyze comparison between two servers."""
    baseline_success = sum(1 for r in baseline_results if r.get("success", False))
    test_success = sum(1 for r in test_results if r.get("success", False))
    # Use max of both result lengths to handle cases where one provider failed completely
    total = max(len(baseline_results), len(test_results), 1)

    baseline_avg_time = sum(
        r.get("execution_time", 0) for r in baseline_results if r.get("success", False)
    ) / max(baseline_success, 1)
    test_avg_time = sum(
        r.get("execution_time", 0) for r in test_results if r.get("success", False)
    ) / max(test_success, 1)

    baseline_avg_iterations = sum(
        r.get("iterations", 0) for r in baseline_results if r.get("success", False)
    ) / max(baseline_success, 1)
    test_avg_iterations = sum(
        r.get("iterations", 0) for r in test_results if r.get("success", False)
    ) / max(test_success, 1)

    # Calculate failed tool calls
    baseline_failed_tools = sum(r.get("failed_tool_calls", 0) for r in baseline_results)
    test_failed_tools = sum(r.get("failed_tool_calls", 0) for r in test_results)

    # Calculate token usage
    baseline_total_tokens = 0
    test_total_tokens = 0

    for r in baseline_results:
        if r.get("token_usage"):
            baseline_total_tokens += r["token_usage"].get("total_tokens", 0)

    for r in test_results:
        if r.get("token_usage"):
            test_total_tokens += r["token_usage"].get("total_tokens", 0)

    return {
        "total_prompts": total,
        "success_rates": {
            baseline: baseline_success / total,
            test: test_success / total,
        },
        "success_counts": {baseline: baseline_success, test: test_success},
        "average_execution_times": {baseline: baseline_avg_time, test: test_avg_time},
        "average_iterations": {
            baseline: baseline_avg_iterations,
            test: test_avg_iterations,
        },
        "total_failed_tool_calls": {
            baseline: baseline_failed_tools,
            test: test_failed_tools,
        },
        "total_token_usage": {baseline: baseline_total_tokens, test: test_total_tokens},
        "winner": {
            "success_rate": (
                "tie"
                if baseline_success == test_success
                else baseline if baseline_success > test_success else test
            ),
            "speed": (
                "tie"
                if baseline_success == 0 and test_success == 0
                else (
                    "tie"
                    if baseline_success > 0
                    and test_success > 0
                    and abs(baseline_avg_time - test_avg_time) < 0.1
                    else (
                        baseline
                        if test_success == 0
                        or (baseline_success > 0 and baseline_avg_time <= test_avg_time)
                        else test
                    )
                )
            ),
            "efficiency": (
                "tie"
                if baseline_success == 0 and test_success == 0
                else (
                    "tie"
                    if baseline_success > 0
                    and test_success > 0
                    and baseline_avg_iterations == test_avg_iterations
                    else (
                        baseline
                        if test_success == 0
                        or (
                            baseline_success > 0
                            and baseline_avg_iterations <= test_avg_iterations
                        )
                        else test
                    )
                )
            ),
            "reliability": (
                "tie"
                if len(baseline_results) == 0 and len(test_results) == 0
                else (
                    "tie"
                    if len(baseline_results) > 0
                    and len(test_results) > 0
                    and baseline_failed_tools == test_failed_tools
                    else (
                        baseline
                        if len(test_results) == 0
                        or (
                            len(baseline_results) > 0
                            and baseline_failed_tools <= test_failed_tools
                        )
                        else test
                    )
                )
            ),
            "token_efficiency": (
                "tie"
                if (baseline_total_tokens or 0) == (test_total_tokens or 0)
                else (
                    baseline
                    if (baseline_total_tokens or 0) < (test_total_tokens or 0)
                    else test
                )
            ),
        },
    }


def _print_comparison_summary(analysis: dict) -> None:
    """Print comparison summary to console."""
    baseline = list(analysis["success_rates"].keys())[0]
    test = list(analysis["success_rates"].keys())[1]

    click.echo(f"\n{'='*70}")
    click.echo("🏆 COMPARISON SUMMARY")
    click.echo(f"{'='*70}")

    click.echo(f"📊 Total prompts: {analysis['total_prompts']}")
    click.echo()

    # Success rates
    click.echo("✅ SUCCESS RATES:")
    baseline_rate = analysis["success_rates"][baseline] * 100
    test_rate = analysis["success_rates"][test] * 100
    click.echo(
        f"  {baseline}: {analysis['success_counts'][baseline]}/{analysis['total_prompts']} ({baseline_rate:.1f}%)"
    )
    click.echo(
        f"  {test}: {analysis['success_counts'][test]}/{analysis['total_prompts']} ({test_rate:.1f}%)"
    )
    click.echo(f"  🏆 Winner: {analysis['winner']['success_rate']}")
    click.echo()

    # Speed
    click.echo("⚡ AVERAGE EXECUTION TIME:")
    baseline_time = analysis["average_execution_times"][baseline]
    test_time = analysis["average_execution_times"][test]
    click.echo(f"  {baseline}: {baseline_time:.2f}s")
    click.echo(f"  {test}: {test_time:.2f}s")
    click.echo(f"  🏆 Winner: {analysis['winner']['speed']}")
    click.echo()

    # Efficiency
    click.echo("🎯 AVERAGE ITERATIONS:")
    baseline_iter = analysis["average_iterations"][baseline]
    test_iter = analysis["average_iterations"][test]
    click.echo(f"  {baseline}: {baseline_iter:.1f}")
    click.echo(f"  {test}: {test_iter:.1f}")
    click.echo(f"  🏆 Winner: {analysis['winner']['efficiency']}")
    click.echo()

    # Reliability (Failed Tool Calls)
    click.echo("🔒 RELIABILITY (Failed Tool Calls):")
    baseline_failed = analysis["total_failed_tool_calls"][baseline]
    test_failed = analysis["total_failed_tool_calls"][test]
    click.echo(f"  {baseline}: {baseline_failed}")
    click.echo(f"  {test}: {test_failed}")
    click.echo(f"  🏆 Winner: {analysis['winner']['reliability']}")
    click.echo()

    # Token efficiency
    click.echo("💎 TOKEN EFFICIENCY:")
    baseline_tokens = analysis["total_token_usage"][baseline]
    test_tokens = analysis["total_token_usage"][test]
    if (baseline_tokens and baseline_tokens > 0) or (test_tokens and test_tokens > 0):
        click.echo(
            f"  {baseline}: {baseline_tokens:,} tokens"
            if baseline_tokens
            else f"  {baseline}: N/A"
        )
        click.echo(
            f"  {test}: {test_tokens:,} tokens" if test_tokens else f"  {test}: N/A"
        )
        click.echo(f"  🏆 Winner: {analysis['winner']['token_efficiency']}")
    else:
        click.echo("  No token usage data available")
    click.echo(f"{'='*70}")


if __name__ == "__main__":
    main()
