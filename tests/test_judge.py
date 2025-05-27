"""Tests for the LLM-as-judge evaluation system."""

from unittest.mock import AsyncMock, Mock

import pytest

from smolval.judge import (
    DEFAULT_CRITERIA,
    JudgmentCriteria,
    JudgmentResult,
    JudgmentScore,
    LLMJudge,
)
from smolval.llm_client import LLMClient, LLMResponse


class TestLLMJudge:
    """Test LLM-as-judge functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = Mock(spec=LLMClient)
        return mock_client

    @pytest.fixture
    def sample_result_data(self):
        """Sample evaluation result data."""
        return {
            "prompt": "List the files in the current directory",
            "result": {
                "success": True,
                "final_answer": "I found 3 files in the current directory: file1.txt, file2.py, and README.md",
                "steps": [
                    {
                        "iteration": 1,
                        "thought": "I need to list the files in the current directory",
                        "action": "list_directory",
                        "action_input": {"path": "."},
                        "observation": "Found files: file1.txt, file2.py, README.md",
                    }
                ],
                "total_iterations": 1,
                "execution_time_seconds": 2.5,
                "failed_tool_calls": 0,
                "error": None,
            },
            "metadata": {
                "config_file": "config/test.yaml",
                "prompt_file": "test_prompt.txt",
                "timestamp": 1234567890.0,
            },
        }

    @pytest.fixture
    def sample_judgment_response(self):
        """Sample LLM judgment response."""
        return """
        {
            "score": 0.85,
            "reasoning": "The answer is accurate and complete. The agent correctly used the list_directory tool and provided a clear summary of the files found.",
            "strengths": ["Clear final answer", "Appropriate tool selection", "Efficient execution"],
            "weaknesses": ["Could have provided more file details"],
            "details": {
                "accuracy": 0.9,
                "completeness": 0.8,
                "clarity": 0.9,
                "relevance": 0.8
            }
        }
        """

    @pytest.mark.asyncio
    async def test_judge_creation(self, mock_llm_client):
        """Test creating an LLM judge."""
        judge = LLMJudge(mock_llm_client)

        assert judge.llm_client == mock_llm_client
        assert len(judge.criteria) == 4  # Default criteria
        assert judge.judge_model is None

    @pytest.mark.asyncio
    async def test_judge_with_custom_criteria(self, mock_llm_client):
        """Test creating a judge with custom criteria."""
        custom_criteria = [
            JudgmentCriteria(
                name="custom_criterion",
                description="A custom test criterion",
                weight=1.0,
                prompt_template="Rate this: {prompt}",
            )
        ]

        judge = LLMJudge(mock_llm_client, criteria=custom_criteria)

        assert len(judge.criteria) == 1
        assert judge.criteria[0].name == "custom_criterion"

    @pytest.mark.asyncio
    async def test_weight_normalization(self, mock_llm_client):
        """Test that criteria weights are normalized to sum to 1.0."""
        custom_criteria = [
            JudgmentCriteria(
                name="criterion1",
                description="First criterion",
                weight=2.0,  # Will be normalized to 0.5
                prompt_template="Test",
            ),
            JudgmentCriteria(
                name="criterion2",
                description="Second criterion",
                weight=2.0,  # Will be normalized to 0.5
                prompt_template="Test",
            ),
        ]

        judge = LLMJudge(mock_llm_client, criteria=custom_criteria)

        total_weight = sum(c.weight for c in judge.criteria)
        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_evaluate_criterion(
        self, mock_llm_client, sample_result_data, sample_judgment_response
    ):
        """Test evaluating a single criterion."""
        # Setup mock response
        mock_response = LLMResponse(content=sample_judgment_response)
        mock_llm_client.generate = AsyncMock(return_value=mock_response)

        judge = LLMJudge(mock_llm_client)
        criterion = DEFAULT_CRITERIA[0]  # answer_quality

        score = await judge._evaluate_criterion(
            criterion, sample_result_data, "Test prompt"
        )

        assert isinstance(score, JudgmentScore)
        assert score.criterion == "answer_quality"
        assert 0.0 <= score.score <= 1.0
        assert len(score.reasoning) > 0

    @pytest.mark.asyncio
    async def test_judge_result_full(self, mock_llm_client, sample_result_data):
        """Test full judgment of a result."""

        # Mock LLM responses for each criterion
        def mock_generate(messages, model=None):
            return LLMResponse(
                content="""{
                "score": 0.8,
                "reasoning": "Good performance overall",
                "strengths": ["Clear answer"],
                "weaknesses": ["Minor improvements possible"],
                "details": {"accuracy": 0.8}
            }"""
            )

        mock_llm_client.generate = AsyncMock(side_effect=mock_generate)

        judge = LLMJudge(mock_llm_client)
        judgment = await judge.judge_result(sample_result_data, "Test prompt")

        assert isinstance(judgment, JudgmentResult)
        assert 0.0 <= judgment.overall_score <= 1.0
        assert len(judgment.scores) == 4  # Number of default criteria
        assert len(judgment.summary) > 0

    @pytest.mark.asyncio
    async def test_judge_with_malformed_response(
        self, mock_llm_client, sample_result_data
    ):
        """Test handling of malformed LLM responses."""
        # Mock malformed JSON response
        mock_response = LLMResponse(content="This is not valid JSON")
        mock_llm_client.generate = AsyncMock(return_value=mock_response)

        judge = LLMJudge(mock_llm_client)

        # Should not raise exception, should provide fallback score
        judgment = await judge.judge_result(sample_result_data, "Test prompt")

        assert isinstance(judgment, JudgmentResult)
        assert 0.0 <= judgment.overall_score <= 1.0

    def test_default_criteria_structure(self):
        """Test that default criteria are properly structured."""
        assert len(DEFAULT_CRITERIA) == 4

        criteria_names = [c.name for c in DEFAULT_CRITERIA]
        expected_names = [
            "answer_quality",
            "reasoning_quality",
            "process_efficiency",
            "task_understanding",
        ]
        assert criteria_names == expected_names

        # Check weights sum to 1.0
        total_weight = sum(c.weight for c in DEFAULT_CRITERIA)
        assert abs(total_weight - 1.0) < 0.01

        # Check all criteria have templates
        for criterion in DEFAULT_CRITERIA:
            assert len(criterion.prompt_template) > 0
            assert "{prompt}" in criterion.prompt_template

    @pytest.mark.asyncio
    async def test_context_preparation(self, mock_llm_client, sample_result_data):
        """Test context preparation for different criteria."""
        judge = LLMJudge(mock_llm_client)

        # Test reasoning quality context
        context = judge._prepare_context(
            "reasoning_quality", sample_result_data, "Test prompt"
        )
        assert "reasoning_steps" in context
        assert "total_iterations" in context

        # Test process efficiency context
        context = judge._prepare_context(
            "process_efficiency", sample_result_data, "Test prompt"
        )
        assert "tool_usage_summary" in context
        assert "execution_time_seconds" in context

        # Test task understanding context
        context = judge._prepare_context(
            "task_understanding", sample_result_data, "Test prompt"
        )
        assert "approach_summary" in context
        assert "final_answer" in context
