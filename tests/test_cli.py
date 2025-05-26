"""Tests for CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from smolval.cli import (
    _analyze_comparison,
    _connect_mcp_servers_silently,
    _print_comparison_summary,
    _show_banner,
    main,
)


class TestCLI:
    """Test CLI functions."""

    def test_analyze_comparison_basic(self):
        """Test basic comparison analysis."""
        baseline_result = {
            "success": True,
            "iterations": 3,
            "execution_time": 5.0,
            "tool_calls_successful": 2,
            "tool_calls_failed": 0,
            "total_tokens": 100,
            "steps": [
                {
                    "step": 1,
                    "thought": "Test thought",
                    "action": {"tool": "test", "arguments": {}},
                    "observation": "Test observation",
                }
            ],
            "final_output": "Test output",
        }

        test_result = {
            "success": True,
            "iterations": 2,
            "execution_time": 3.0,
            "tool_calls_successful": 2,
            "tool_calls_failed": 1,
            "total_tokens": 80,
            "steps": [
                {
                    "step": 1,
                    "thought": "Test thought",
                    "action": {"tool": "test", "arguments": {}},
                    "observation": "Test observation",
                }
            ],
            "final_output": "Test output",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        assert analysis["success_rates"]["baseline"] == 1.0
        assert analysis["success_rates"]["test"] == 1.0
        assert analysis["total_prompts"] == 1
        assert "winner" in analysis
        assert "average_execution_times" in analysis
        assert "total_token_usage" in analysis

    def test_analyze_comparison_with_failures(self):
        """Test comparison analysis with failures."""
        baseline_result = {
            "success": False,
            "iterations": 5,
            "execution_time": 10.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 2,
            "total_tokens": 200,
            "steps": [],
            "final_output": "Failed",
        }

        test_result = {
            "success": True,
            "iterations": 2,
            "execution_time": 3.0,
            "tool_calls_successful": 3,
            "tool_calls_failed": 0,
            "total_tokens": 80,
            "steps": [],
            "final_output": "Success",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        assert analysis["success_rates"]["baseline"] == 0.0
        assert analysis["success_rates"]["test"] == 1.0
        assert analysis["total_prompts"] == 1

    def test_analyze_comparison_edge_cases(self):
        """Test comparison analysis with edge cases."""
        # Test with zero tokens
        baseline_result = {
            "success": True,
            "iterations": 1,
            "execution_time": 1.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 0,
            "total_tokens": 0,
            "steps": [],
            "final_output": "Success",
        }

        test_result = {
            "success": True,
            "iterations": 1,
            "execution_time": 1.0,
            "tool_calls_successful": 1,
            "tool_calls_failed": 0,
            "total_tokens": 100,
            "steps": [],
            "final_output": "Success",
        }

        analysis = _analyze_comparison(
            "baseline", "test", [baseline_result], [test_result]
        )

        # Should handle division by zero gracefully
        assert "total_token_usage" in analysis
        assert "baseline" in analysis["total_token_usage"]
        assert "test" in analysis["total_token_usage"]

    def test_show_banner(self):
        """Test showing ASCII banner."""
        with patch('smolval.cli.click.echo') as mock_echo:
            _show_banner()
            
            # Should call click.echo with styled banner
            mock_echo.assert_called_once()
            args = mock_echo.call_args[0]
            assert len(args) == 1
            # Banner should contain ASCII art
            banner_text = str(args[0])
            assert "smolval" in banner_text.lower() or "███" in banner_text

    @pytest.mark.asyncio
    async def test_connect_mcp_servers_silently_debug(self):
        """Test connecting MCP servers with debug mode."""
        mock_manager = AsyncMock()
        server_configs = [
            {"name": "test1", "command": ["echo", "test"]},
            {"name": "test2", "command": ["echo", "test"]}
        ]
        
        await _connect_mcp_servers_silently(mock_manager, server_configs, debug=True)
        
        # Should call connect for each server with debug=True
        assert mock_manager.connect.call_count == 2
        for call in mock_manager.connect.call_args_list:
            assert call[1]["debug"] is True

    @pytest.mark.asyncio
    async def test_connect_mcp_servers_silently_no_debug(self):
        """Test connecting MCP servers with stderr suppression."""
        mock_manager = AsyncMock()
        server_configs = [{"name": "test", "command": ["echo", "test"]}]
        
        with patch('smolval.cli.os.dup') as mock_dup, \
             patch('smolval.cli.os.open') as mock_open, \
             patch('smolval.cli.os.dup2') as mock_dup2, \
             patch('smolval.cli.os.close') as mock_close:
            
            mock_dup.return_value = 999
            mock_open.return_value = 888
            
            await _connect_mcp_servers_silently(mock_manager, server_configs, debug=False)
            
            # Should call connect with debug=False
            mock_manager.connect.assert_called_once()
            assert mock_manager.connect.call_args[1]["debug"] is False
            
            # Should manage file descriptors for stderr suppression
            mock_dup.assert_called()
            mock_open.assert_called()
            mock_dup2.assert_called()
            mock_close.assert_called()

    def test_print_comparison_summary(self):
        """Test printing comparison summary."""
        analysis = {
            "baseline_server": "server1",
            "test_server": "server2", 
            "total_prompts": 10,
            "success_rates": {"server1": 0.8, "server2": 0.9},
            "success_counts": {"server1": 8, "server2": 9},
            "average_execution_times": {"server1": 5.0, "server2": 3.0},
            "average_iterations": {"server1": 3.5, "server2": 2.5},
            "tool_call_failures": {"server1": 1, "server2": 0},
            "total_token_usage": {"server1": 1000, "server2": 800},
            "winner": {
                "success_rate": "server2",
                "speed": "server2", 
                "iterations": "server2"
            }
        }
        
        with patch('smolval.cli.click.echo') as mock_echo:
            _print_comparison_summary(analysis)
            
            # Should print comparison details
            assert mock_echo.call_count >= 5  # Multiple echo calls for different sections
            
            # Check that important information is printed
            all_output = " ".join(str(call[0][0]) for call in mock_echo.call_args_list)
            assert "server1" in all_output
            assert "server2" in all_output
            assert "80.0%" in all_output or "0.8" in all_output
            assert "90.0%" in all_output or "0.9" in all_output

    def test_print_comparison_summary_no_tokens(self):
        """Test printing comparison summary without token data."""
        analysis = {
            "baseline_server": "server1",
            "test_server": "server2",
            "total_prompts": 5,
            "success_rates": {"server1": 1.0, "server2": 1.0},
            "success_counts": {"server1": 5, "server2": 5},
            "average_execution_times": {"server1": 5.0, "server2": 3.0},
            "average_iterations": {"server1": 3.0, "server2": 2.0},
            "tool_call_failures": {"server1": 0, "server2": 0},
            "total_token_usage": {"server1": None, "server2": None},
            "winner": {
                "success_rate": "tie",
                "speed": "server2",
                "iterations": "server2"
            }
        }
        
        with patch('smolval.cli.click.echo') as mock_echo:
            _print_comparison_summary(analysis)
            
            # Should handle missing token data gracefully
            assert mock_echo.call_count >= 3
            all_output = " ".join(str(call[0][0]) for call in mock_echo.call_args_list)
            # Should not crash and should still show other metrics
            assert "server1" in all_output
            assert "server2" in all_output


class TestMainCommand:
    """Test main CLI command and entry points."""

    def test_main_help(self):
        """Test main command help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "smolval" in result.output.lower()
        assert "eval" in result.output
        assert "batch" in result.output
        assert "compare" in result.output

    def test_main_version(self):
        """Test main command version output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_main_verbose_flag(self):
        """Test main command with verbose flag."""
        runner = CliRunner()
        # Test with a command that will trigger main callback, not --help
        with patch('smolval.cli.logging.basicConfig') as mock_logging:
            # Use a command that exists but will fail (missing file)
            result = runner.invoke(main, ['--verbose', 'eval', 'nonexistent.txt'])
            
            # Should configure logging with INFO level  
            mock_logging.assert_called_once()
            args = mock_logging.call_args[1]
            assert args['level'] == 20  # logging.INFO

    def test_main_debug_flag(self):
        """Test main command with debug flag."""
        runner = CliRunner()
        with patch('smolval.cli.logging.basicConfig') as mock_logging:
            # Use a command that exists but will fail (missing file)
            result = runner.invoke(main, ['--debug', 'eval', 'nonexistent.txt'])
            
            # Should configure logging with DEBUG level
            mock_logging.assert_called_once()
            args = mock_logging.call_args[1]
            assert args['level'] == 10  # logging.DEBUG

    def test_eval_command_missing_file(self):
        """Test eval command with missing prompt file."""
        runner = CliRunner()
        result = runner.invoke(main, ['eval', 'nonexistent.txt'])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output or "No such file" in result.output

    def test_eval_command_help(self):
        """Test eval command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['eval', '--help'])
        
        assert result.exit_code == 0
        assert "prompt" in result.output.lower()
        assert "config" in result.output.lower()

    def test_batch_command_missing_dir(self):
        """Test batch command with missing directory."""
        runner = CliRunner()
        result = runner.invoke(main, ['batch', 'nonexistent_dir'])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output or "No such file" in result.output

    def test_batch_command_help(self):
        """Test batch command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['batch', '--help'])
        
        assert result.exit_code == 0
        assert "prompts" in result.output.lower()
        assert "servers" in result.output.lower()

    def test_compare_command_help(self):
        """Test compare command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['compare', '--help'])
        
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()
        assert "test" in result.output.lower()

    def test_compare_command_missing_args(self):
        """Test compare command with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ['compare', 'some_dir'])
        
        assert result.exit_code != 0
        # Should complain about missing --baseline and --test

    @patch('smolval.cli._run_eval')
    def test_eval_command_with_valid_file(self, mock_run_eval):
        """Test eval command with valid file."""
        runner = CliRunner()
        
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test prompt content")
            prompt_file = f.name
        
        try:
            # Mock the async function to return successfully
            mock_run_eval.return_value = None
            
            result = runner.invoke(main, ['--no-banner', 'eval', prompt_file])
            
            # Should call the async function
            mock_run_eval.assert_called_once()
            
        finally:
            # Cleanup
            Path(prompt_file).unlink()

    @patch('smolval.cli._run_batch')
    def test_batch_command_with_valid_dir(self, mock_run_batch):
        """Test batch command with valid directory."""
        runner = CliRunner()
        
        # Create a temporary directory with prompt file
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_file = Path(temp_dir) / "test.txt"
            prompt_file.write_text("Test prompt")
            
            # Mock the async function
            mock_run_batch.return_value = None
            
            result = runner.invoke(main, ['--no-banner', 'batch', temp_dir])
            
            # Should call the async function
            mock_run_batch.assert_called_once()

    def test_no_banner_flag(self):
        """Test --no-banner flag prevents banner display."""
        runner = CliRunner()
        
        with patch('smolval.cli._show_banner') as mock_banner:
            result = runner.invoke(main, ['--no-banner', '--help'])
            
            # Banner should not be called
            mock_banner.assert_not_called()

    def test_banner_displayed_by_default(self):
        """Test banner is displayed by default for commands."""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test prompt")
            prompt_file = f.name
        
        try:
            with patch('smolval.cli._show_banner') as mock_banner, \
                 patch('smolval.cli._run_eval') as mock_run_eval:
                
                mock_run_eval.return_value = None
                result = runner.invoke(main, ['eval', prompt_file])
                
                # Banner should be called
                mock_banner.assert_called_once()
                
        finally:
            Path(prompt_file).unlink()
