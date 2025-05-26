"""Tests for output_manager module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from smolval.output_manager import OutputManager


class TestOutputManager:
    """Test OutputManager class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        manager = OutputManager()
        assert manager.base_output_dir == Path("results")
        assert manager.current_run_dir is None

    def test_init_custom_dir(self):
        """Test initialization with custom base directory."""
        manager = OutputManager("custom_results")
        assert manager.base_output_dir == Path("custom_results")
        assert manager.current_run_dir is None

    def test_create_run_directory_no_name(self):
        """Test creating run directory without name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                
                run_dir = manager.create_run_directory()
                
                expected_path = Path(temp_dir) / "20240115_143000"
                assert run_dir == expected_path
                assert run_dir.exists()
                assert manager.current_run_dir == expected_path

    def test_create_run_directory_with_name(self):
        """Test creating run directory with custom name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                
                run_dir = manager.create_run_directory("test_eval")
                
                expected_path = Path(temp_dir) / "20240115_143000_test_eval"
                assert run_dir == expected_path
                assert run_dir.exists()
                assert manager.current_run_dir == expected_path

    def test_create_run_directory_sanitize_name(self):
        """Test creating run directory with name that needs sanitization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                
                # Test various special characters that should be stripped
                run_dir = manager.create_run_directory("test/eval:with*bad?chars|<>")
                
                expected_path = Path(temp_dir) / "20240115_143000_testevalwithbadchars"
                assert run_dir == expected_path
                assert run_dir.exists()

    def test_create_run_directory_empty_name_after_sanitize(self):
        """Test creating run directory when name becomes empty after sanitization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                
                # Name with only special characters
                run_dir = manager.create_run_directory("!@#$%^&*()")
                
                expected_path = Path(temp_dir) / "20240115_143000_"
                assert run_dir == expected_path
                assert run_dir.exists()

    def test_get_run_directory_existing(self):
        """Test getting run directory when already created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            # Create directory first
            created_dir = manager.create_run_directory("test")
            
            # Get should return same directory
            retrieved_dir = manager.get_run_directory()
            assert retrieved_dir == created_dir
            assert retrieved_dir == manager.current_run_dir

    def test_get_run_directory_creates_if_none(self):
        """Test getting run directory creates one if none exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                
                # Should create directory since none exists
                run_dir = manager.get_run_directory()
                
                expected_path = Path(temp_dir) / "20240115_143000"
                assert run_dir == expected_path
                assert run_dir.exists()
                assert manager.current_run_dir == expected_path

    @patch('smolval.output_manager.ResultsFormatter')
    def test_write_evaluation_results_no_name(self, mock_formatter_class):
        """Test writing evaluation results without eval name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            # Setup mock formatter
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_single_result.return_value = "formatted_result"
            
            # Create run directory
            manager.create_run_directory("test")
            
            result_data = {"success": True, "test": "data"}
            
            # Write results
            files = manager.write_evaluation_results(result_data)
            
            # Check that formatter was created for each format (json, markdown, html)
            assert mock_formatter_class.call_count == 3
            
            # Check that format_single_result was called for each format
            assert mock_formatter.format_single_result.call_count == 3
            
            # Check return value structure
            assert "json" in files
            assert "markdown" in files
            assert "html" in files
            
            # Check filenames contain timestamp-based eval name
            assert files["json"].endswith(".json")
            assert files["markdown"].endswith(".md")
            assert files["html"].endswith(".html")
            assert "eval_" in files["json"]

    @patch('smolval.output_manager.ResultsFormatter')
    def test_write_evaluation_results_with_name(self, mock_formatter_class):
        """Test writing evaluation results with eval name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            # Setup mock formatter
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_single_result.return_value = "formatted_result"
            
            # Create run directory
            manager.create_run_directory("test")
            
            result_data = {"success": True, "test": "data"}
            
            # Write results with custom name
            files = manager.write_evaluation_results(result_data, "custom_eval")
            
            # Check filenames include custom name
            assert files["json"].endswith("eval_custom_eval.json")
            assert files["markdown"].endswith("eval_custom_eval.md")
            assert files["html"].endswith("eval_custom_eval.html")

    @patch('smolval.output_manager.ResultsFormatter')
    def test_write_batch_results(self, mock_formatter_class):
        """Test writing batch results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            # Setup mock formatter
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_batch_results.return_value = "formatted_batch"
            
            manager.create_run_directory("test")
            
            batch_data = {"batch_name": "test", "results": []}
            
            files = manager.write_batch_results(batch_data, "test_batch")
            
            # Check that formatter was created for each format (json, markdown, html)
            assert mock_formatter_class.call_count == 3
            
            # Check that format_batch_results was called for each format
            assert mock_formatter.format_batch_results.call_count == 3
            
            # Check return value structure
            assert "json" in files
            assert "markdown" in files
            assert "html" in files
            
            # Check filenames
            assert files["json"].endswith("batch_test_batch.json")

    @patch('smolval.output_manager.ResultsFormatter')
    def test_write_comparison_results(self, mock_formatter_class):
        """Test writing comparison results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            # Setup mock formatter
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_comparison_results.return_value = "formatted_comparison"
            
            manager.create_run_directory("test")
            
            comparison_data = {"baseline": "server1", "test": "server2"}
            
            files = manager.write_comparison_results(comparison_data, "server_comparison")
            
            # Check that formatter was created for each format (json, markdown, html)
            assert mock_formatter_class.call_count == 3
            
            # Check that format_comparison_results was called for each format
            assert mock_formatter.format_comparison_results.call_count == 3
            
            # Check filenames
            assert files["json"].endswith("comparison_server_comparison.json")

    def test_create_nested_directories(self):
        """Test creating nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use nested path
            nested_path = Path(temp_dir) / "deep" / "nested" / "structure"
            manager = OutputManager(str(nested_path))
            
            run_dir = manager.create_run_directory("test")
            
            # Should create all parent directories
            assert run_dir.exists()
            assert run_dir.parent == nested_path  # run_dir is nested_path/timestamp_test

    def test_multiple_run_directories(self):
        """Test creating multiple run directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(temp_dir)
            
            with patch('smolval.output_manager.datetime') as mock_datetime:
                # First directory
                mock_datetime.now.return_value.strftime.return_value = "20240115_143000"
                dir1 = manager.create_run_directory("run1")
                
                # Second directory (simulates time passing)
                mock_datetime.now.return_value.strftime.return_value = "20240115_143001"
                dir2 = manager.create_run_directory("run2")
                
                # Both should exist and be different
                assert dir1.exists()
                assert dir2.exists()
                assert dir1 != dir2
                assert manager.current_run_dir == dir2  # Should be latest