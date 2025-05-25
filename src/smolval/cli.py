"""Command-line interface for smolval."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

import click

from .config import Config
from .llm_client import LLMClient
from .mcp_client import MCPClientManager
from .agent import Agent
from .results import ResultsFormatter


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """smolval: A lightweight MCP server evaluation agent."""
    pass


@main.command()
@click.argument('prompt_file', type=click.Path(exists=True))
@click.option('--config', '-c', default='config/no-api-keys.yaml', 
              help='Configuration file path')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv', 'markdown', 'html']),
              help='Output format')
def eval(prompt_file: str, config: str, output: str, output_format: str) -> None:
    """Evaluate MCP servers using a prompt file."""
    asyncio.run(_run_eval(prompt_file, config, output, output_format))


async def _run_eval(prompt_file: str, config_path: str, output: str, output_format: str) -> None:
    """Run evaluation asynchronously."""
    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        config = Config.from_yaml(Path(config_path))
        
        # Load prompt
        click.echo(f"Loading prompt from: {prompt_file}")
        with open(prompt_file, 'r') as f:
            prompt = f.read().strip()
        
        # Initialize components
        click.echo("Initializing LLM client...")
        llm_client = LLMClient(config.llm)
        
        click.echo("Initializing MCP client manager...")
        mcp_manager = MCPClientManager()
        
        # Connect to MCP servers
        click.echo("Connecting to MCP servers...")
        for server_config in config.mcp_servers:
            await mcp_manager.connect(server_config)
        
        # Initialize agent
        click.echo("Initializing agent...")
        agent = Agent(config, llm_client, mcp_manager)
        
        # Run evaluation
        click.echo("Running evaluation...")
        start_time = time.time()
        result = await agent.run(prompt)
        end_time = time.time()
        
        # Format result
        result_data = {
            "prompt": prompt,
            "result": {
                "success": result.success,
                "final_answer": result.final_answer,
                "steps": [step.model_dump() for step in result.steps],
                "total_iterations": result.total_iterations,
                "error": result.error,
                "execution_time_seconds": result.execution_time_seconds
            },
            "metadata": {
                "config_file": config_path,
                "prompt_file": prompt_file,
                "timestamp": time.time(),
                "duration_seconds": end_time - start_time
            }
        }
        
        # Output result
        formatter = ResultsFormatter(output_format)
        if output:
            click.echo(f"Writing results to: {output}")
            formatter.format_single_result(result_data, output)
        else:
            # Pretty print to console
            if output_format == "json":
                _pretty_print_result(result_data)
            else:
                formatted_output = formatter.format_single_result(result_data)
                click.echo(formatted_output)
        
        # Clean up
        await mcp_manager.disconnect_all()
        
        if result.success:
            click.echo("✅ Evaluation completed successfully")
        else:
            click.echo("❌ Evaluation failed")
            if result.error:
                click.echo(f"Error: {result.error}")
                
    except Exception as e:
        click.echo(f"❌ Evaluation failed with error: {e}")
        raise click.ClickException(str(e))


def _pretty_print_result(result_data: Dict[str, Any]) -> None:
    """Pretty print evaluation results to console."""
    result = result_data["result"]
    
    click.echo("\n" + "="*60)
    click.echo("EVALUATION RESULTS")
    click.echo("="*60)
    
    click.echo(f"Success: {'✅' if result['success'] else '❌'}")
    click.echo(f"Iterations: {result['total_iterations']}")
    click.echo(f"Execution Time: {result['execution_time_seconds']:.2f}s")
    
    if result['error']:
        click.echo(f"Error: {result['error']}")
    
    click.echo(f"\nFinal Answer:\n{result['final_answer']}")
    
    if result['steps']:
        click.echo(f"\nStep-by-step execution:")
        for i, step in enumerate(result['steps'], 1):
            click.echo(f"\n--- Step {i} (Iteration {step['iteration']}) ---")
            click.echo(f"Thought: {step['thought']}")
            if step['action']:
                click.echo(f"Action: {step['action']}")
                click.echo(f"Action Input: {json.dumps(step['action_input'], indent=2)}")
            if step['observation']:
                click.echo(f"Observation: {step['observation']}")
    
    click.echo("\n" + "="*60)


@main.command()
@click.argument('prompts_dir', type=click.Path(exists=True))
@click.option('--config', '-c', default='config/no-api-keys.yaml', 
              help='Configuration file path')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--servers', help='Comma-separated list of server names to filter')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv', 'markdown', 'html']),
              help='Output format')
def batch(prompts_dir: str, config: str, output: str, servers: str, output_format: str) -> None:
    """Run batch evaluation against multiple prompts."""
    asyncio.run(_run_batch(prompts_dir, config, output, servers, output_format))


async def _run_batch(prompts_dir: str, config_path: str, output_dir: str, servers_filter: str, output_format: str) -> None:
    """Run batch evaluation asynchronously."""
    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        config = Config.from_yaml(Path(config_path))
        
        # Filter servers if specified
        if servers_filter:
            server_names = [s.strip() for s in servers_filter.split(',')]
            config.mcp_servers = [s for s in config.mcp_servers if s.name in server_names]
            click.echo(f"Filtering to servers: {server_names}")
        
        # Find all prompt files
        prompts_path = Path(prompts_dir)
        prompt_files = list(prompts_path.glob("*.txt")) + list(prompts_path.glob("*.md"))
        
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
        
        # Connect to MCP servers
        click.echo("Connecting to MCP servers...")
        for server_config in config.mcp_servers:
            await mcp_manager.connect(server_config)
        
        # Initialize agent
        agent = Agent(config, llm_client, mcp_manager)
        
        # Run evaluations
        results = []
        for i, prompt_file in enumerate(prompt_files, 1):
            click.echo(f"\n[{i}/{len(prompt_files)}] Evaluating: {prompt_file.name}")
            
            try:
                # Load prompt
                with open(prompt_file, 'r') as f:
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
                        "execution_time_seconds": result.execution_time_seconds
                    },
                    "metadata": {
                        "timestamp": time.time(),
                        "duration_seconds": end_time - start_time
                    }
                }
                
                results.append(result_data)
                
                # Write individual result if output directory specified
                if output_dir:
                    output_file = output_path / f"{prompt_file.stem}_result.json"
                    with open(output_file, 'w') as f:
                        json.dump(result_data, f, indent=2)
                
                status = "✅" if result.success else "❌"
                click.echo(f"   {status} {result.execution_time_seconds:.2f}s")
                
            except Exception as e:
                click.echo(f"   ❌ Error: {e}")
                results.append({
                    "prompt_file": str(prompt_file.name),
                    "prompt": "",
                    "result": {
                        "success": False,
                        "final_answer": "",
                        "steps": [],
                        "total_iterations": 0,
                        "error": str(e),
                        "execution_time_seconds": 0
                    },
                    "metadata": {"timestamp": time.time(), "duration_seconds": 0}
                })
        
        # Write summary results
        summary = {
            "config_file": config_path,
            "prompts_directory": prompts_dir,
            "total_prompts": len(prompt_files),
            "successful": sum(1 for r in results if r["result"]["success"]),
            "failed": sum(1 for r in results if not r["result"]["success"]),
            "results": results,
            "metadata": {
                "timestamp": time.time(),
                "servers": [s.name for s in config.mcp_servers]
            }
        }
        
        if output_dir:
            formatter = ResultsFormatter(output_format)
            summary_file = output_path / f"batch_summary.{output_format}"
            formatter.format_batch_results(summary, str(summary_file))
            click.echo(f"\nBatch summary written to: {summary_file}")
        
        # Print summary
        click.echo(f"\n{'='*60}")
        click.echo("BATCH EVALUATION SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total prompts: {summary['total_prompts']}")
        click.echo(f"Successful: {summary['successful']} ✅")
        click.echo(f"Failed: {summary['failed']} ❌")
        click.echo(f"Success rate: {summary['successful']/summary['total_prompts']*100:.1f}%")
        
        # Clean up
        await mcp_manager.disconnect_all()
        
    except Exception as e:
        click.echo(f"❌ Batch evaluation failed: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.option('--baseline', required=True, help='Baseline server name')
@click.option('--test', required=True, help='Test server name')
@click.argument('prompts_dir', type=click.Path(exists=True))
@click.option('--config', '-c', default='config/no-api-keys.yaml', 
              help='Configuration file path')
@click.option('--output', '-o', help='Output file for comparison results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv', 'markdown', 'html']),
              help='Output format')
def compare(baseline: str, test: str, prompts_dir: str, config: str, output: str, output_format: str) -> None:
    """Compare performance between two MCP servers."""
    asyncio.run(_run_compare(baseline, test, prompts_dir, config, output, output_format))


async def _run_compare(baseline: str, test: str, prompts_dir: str, config_path: str, output: str, output_format: str) -> None:
    """Run comparison asynchronously."""
    try:
        # Load configuration
        click.echo(f"Loading config from: {config_path}")
        config = Config.from_yaml(Path(config_path))
        
        # Validate servers exist in config
        server_names = [s.name for s in config.mcp_servers]
        if baseline not in server_names:
            raise click.ClickException(f"Baseline server '{baseline}' not found in config")
        if test not in server_names:
            raise click.ClickException(f"Test server '{test}' not found in config")
        
        # Find all prompt files
        prompts_path = Path(prompts_dir)
        prompt_files = list(prompts_path.glob("*.txt")) + list(prompts_path.glob("*.md"))
        
        if not prompt_files:
            raise click.ClickException(f"No prompt files found in {prompts_dir}")
        
        click.echo(f"Found {len(prompt_files)} prompt files")
        click.echo(f"Comparing: {baseline} vs {test}")
        
        # Run evaluations for both servers
        comparison_results = []
        
        for server_name in [baseline, test]:
            click.echo(f"\n--- Running evaluations for {server_name} ---")
            
            # Filter config to only this server
            server_config = Config(
                mcp_servers=[s for s in config.mcp_servers if s.name == server_name],
                llm=config.llm,
                evaluation=config.evaluation
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
                    with open(prompt_file, 'r') as f:
                        prompt = f.read().strip()
                    
                    # Run evaluation
                    start_time = time.time()
                    result = await agent.run(prompt)
                    end_time = time.time()
                    
                    server_results.append({
                        "prompt_file": str(prompt_file.name),
                        "prompt": prompt,
                        "success": result.success,
                        "final_answer": result.final_answer,
                        "iterations": result.total_iterations,
                        "execution_time": result.execution_time_seconds,
                        "error": result.error,
                        "steps": len(result.steps)
                    })
                    
                    status = "✅" if result.success else "❌"
                    click.echo(f"    {status} {result.execution_time_seconds:.2f}s")
                    
                except Exception as e:
                    click.echo(f"    ❌ Error: {e}")
                    server_results.append({
                        "prompt_file": str(prompt_file.name),
                        "prompt": "",
                        "success": False,
                        "final_answer": "",
                        "iterations": 0,
                        "execution_time": 0,
                        "error": str(e),
                        "steps": 0
                    })
            
            comparison_results.append({
                "server_name": server_name,
                "results": server_results
            })
            
            # Clean up
            await mcp_manager.disconnect_all()
        
        # Generate comparison analysis
        baseline_results = comparison_results[0]["results"]
        test_results = comparison_results[1]["results"]
        
        analysis = _analyze_comparison(baseline, test, baseline_results, test_results)
        
        # Create final comparison data
        comparison_data = {
            "baseline_server": baseline,
            "test_server": test,
            "prompts_directory": prompts_dir,
            "config_file": config_path,
            "analysis": analysis,
            "detailed_results": {
                baseline: baseline_results,
                test: test_results
            },
            "metadata": {
                "timestamp": time.time(),
                "total_prompts": len(prompt_files)
            }
        }
        
        # Output results
        if output:
            click.echo(f"\nWriting comparison to: {output}")
            formatter = ResultsFormatter(output_format)
            formatter.format_comparison_results(comparison_data, output)
        
        # Print comparison summary
        _print_comparison_summary(analysis)
        
    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}")
        raise click.ClickException(str(e))


def _analyze_comparison(baseline: str, test: str, baseline_results: list, test_results: list) -> dict:
    """Analyze comparison between two servers."""
    baseline_success = sum(1 for r in baseline_results if r["success"])
    test_success = sum(1 for r in test_results if r["success"])
    total = len(baseline_results)
    
    baseline_avg_time = sum(r["execution_time"] for r in baseline_results if r["success"]) / max(baseline_success, 1)
    test_avg_time = sum(r["execution_time"] for r in test_results if r["success"]) / max(test_success, 1)
    
    baseline_avg_iterations = sum(r["iterations"] for r in baseline_results if r["success"]) / max(baseline_success, 1)
    test_avg_iterations = sum(r["iterations"] for r in test_results if r["success"]) / max(test_success, 1)
    
    return {
        "total_prompts": total,
        "success_rates": {
            baseline: baseline_success / total,
            test: test_success / total
        },
        "success_counts": {
            baseline: baseline_success,
            test: test_success
        },
        "average_execution_times": {
            baseline: baseline_avg_time,
            test: test_avg_time
        },
        "average_iterations": {
            baseline: baseline_avg_iterations,
            test: test_avg_iterations
        },
        "winner": {
            "success_rate": baseline if baseline_success >= test_success else test,
            "speed": baseline if baseline_avg_time <= test_avg_time else test,
            "efficiency": baseline if baseline_avg_iterations <= test_avg_iterations else test
        }
    }


def _print_comparison_summary(analysis: dict) -> None:
    """Print comparison summary to console."""
    baseline = [k for k in analysis["success_rates"].keys()][0]
    test = [k for k in analysis["success_rates"].keys()][1]
    
    click.echo(f"\n{'='*60}")
    click.echo("COMPARISON SUMMARY")
    click.echo(f"{'='*60}")
    
    click.echo(f"Total prompts: {analysis['total_prompts']}")
    click.echo()
    
    # Success rates
    click.echo("SUCCESS RATES:")
    baseline_rate = analysis["success_rates"][baseline] * 100
    test_rate = analysis["success_rates"][test] * 100
    click.echo(f"  {baseline}: {analysis['success_counts'][baseline]}/{analysis['total_prompts']} ({baseline_rate:.1f}%)")
    click.echo(f"  {test}: {analysis['success_counts'][test]}/{analysis['total_prompts']} ({test_rate:.1f}%)")
    click.echo(f"  Winner: {analysis['winner']['success_rate']} 🏆")
    click.echo()
    
    # Speed
    click.echo("AVERAGE EXECUTION TIME:")
    baseline_time = analysis["average_execution_times"][baseline]
    test_time = analysis["average_execution_times"][test]
    click.echo(f"  {baseline}: {baseline_time:.2f}s")
    click.echo(f"  {test}: {test_time:.2f}s")
    click.echo(f"  Winner: {analysis['winner']['speed']} 🏆")
    click.echo()
    
    # Efficiency
    click.echo("AVERAGE ITERATIONS:")
    baseline_iter = analysis["average_iterations"][baseline]
    test_iter = analysis["average_iterations"][test]
    click.echo(f"  {baseline}: {baseline_iter:.1f}")
    click.echo(f"  {test}: {test_iter:.1f}")
    click.echo(f"  Winner: {analysis['winner']['efficiency']} 🏆")
    
    click.echo(f"\n{'='*60}")


if __name__ == '__main__':
    main()