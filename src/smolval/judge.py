"""LLM-as-judge evaluation system for smolval results."""

import json
import logging
from typing import Any

from pydantic import BaseModel

from smolval.llm_client import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


class JudgmentScore(BaseModel):
    """Individual judgment score with reasoning."""
    
    criterion: str
    score: float  # 0.0 to 1.0
    reasoning: str
    details: dict[str, Any] | None = None


class JudgmentResult(BaseModel):
    """Complete judgment result for an evaluation."""
    
    overall_score: float  # 0.0 to 1.0, weighted average
    scores: list[JudgmentScore]
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]


class JudgmentCriteria(BaseModel):
    """Configuration for judgment criteria."""
    
    name: str
    description: str
    weight: float  # Relative weight for overall score
    prompt_template: str


# Default evaluation criteria
DEFAULT_CRITERIA = [
    JudgmentCriteria(
        name="answer_quality",
        description="Accuracy, completeness, and relevance of the final answer",
        weight=0.35,
        prompt_template="""
Evaluate the ANSWER QUALITY of this evaluation result:

**Evaluation Criteria:**
- Accuracy: Is the final answer factually correct and relevant to the task?
- Completeness: Does the answer address all aspects of the prompt/question?
- Clarity: Is the answer well-structured and easy to understand?
- Relevance: Does the answer stay focused on the task requirements?

**Task Prompt:**
{prompt}

**Final Answer:**
{final_answer}

**Success Status:** {success}

Rate the answer quality from 0.0 (completely inadequate) to 1.0 (excellent).

Provide your evaluation in this JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "details": {{
        "accuracy": <0.0-1.0>,
        "completeness": <0.0-1.0>,
        "clarity": <0.0-1.0>,
        "relevance": <0.0-1.0>
    }}
}}
""".strip()
    ),
    
    JudgmentCriteria(
        name="reasoning_quality",
        description="Quality of the step-by-step reasoning process",
        weight=0.25,
        prompt_template="""
Evaluate the REASONING QUALITY of this evaluation result:

**Evaluation Criteria:**
- Logical Flow: Do the thoughts and actions follow a logical sequence?
- Problem Decomposition: Is the task broken down appropriately?
- Tool Selection: Are the chosen tools appropriate for each subtask?
- Adaptability: Does the agent adapt well when tools fail or provide unexpected results?

**Task Prompt:**
{prompt}

**Reasoning Steps:**
{reasoning_steps}

**Total Iterations:** {total_iterations}
**Success Status:** {success}

Rate the reasoning quality from 0.0 (poor reasoning) to 1.0 (excellent reasoning).

Provide your evaluation in this JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "details": {{
        "logical_flow": <0.0-1.0>,
        "problem_decomposition": <0.0-1.0>,
        "tool_selection": <0.0-1.0>,
        "adaptability": <0.0-1.0>
    }}
}}
""".strip()
    ),
    
    JudgmentCriteria(
        name="process_efficiency",
        description="Efficiency and optimization of the execution process",
        weight=0.20,
        prompt_template="""
Evaluate the PROCESS EFFICIENCY of this evaluation result:

**Evaluation Criteria:**
- Tool Usage: Are tools used efficiently without redundancy?
- Information Gathering: Is information collected systematically and completely?
- Error Handling: Are errors and failed tool calls handled appropriately?
- Resource Utilization: Is the process completed with reasonable resource usage?

**Task Prompt:**
{prompt}

**Execution Summary:**
- Total Iterations: {total_iterations}
- Execution Time: {execution_time_seconds:.2f}s
- Failed Tool Calls: {failed_tool_calls}
- Success: {success}

**Tool Usage Pattern:**
{tool_usage_summary}

Rate the process efficiency from 0.0 (very inefficient) to 1.0 (highly efficient).

Provide your evaluation in this JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "details": {{
        "tool_efficiency": <0.0-1.0>,
        "information_gathering": <0.0-1.0>,
        "error_handling": <0.0-1.0>,
        "resource_utilization": <0.0-1.0>
    }}
}}
""".strip()
    ),
    
    JudgmentCriteria(
        name="task_understanding",
        description="Understanding and adherence to task requirements",
        weight=0.20,
        prompt_template="""
Evaluate the TASK UNDERSTANDING of this evaluation result:

**Evaluation Criteria:**
- Prompt Comprehension: Does the agent correctly understand the task requirements?
- Goal Alignment: Are the actions aligned with the intended goal?
- Context Awareness: Does the agent maintain awareness of the task context throughout?
- Requirement Fulfillment: Are all stated and implied requirements addressed?

**Task Prompt:**
{prompt}

**Agent's Approach:**
{approach_summary}

**Final Answer:**
{final_answer}

**Success Status:** {success}

Rate the task understanding from 0.0 (poor understanding) to 1.0 (excellent understanding).

Provide your evaluation in this JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "details": {{
        "prompt_comprehension": <0.0-1.0>,
        "goal_alignment": <0.0-1.0>,
        "context_awareness": <0.0-1.0>,
        "requirement_fulfillment": <0.0-1.0>
    }}
}}
""".strip()
    ),
]


class LLMJudge:
    """LLM-based evaluation system for smolval results."""
    
    def __init__(
        self, 
        llm_client: LLMClient,
        criteria: list[JudgmentCriteria] | None = None,
        judge_model: str | None = None
    ) -> None:
        """Initialize the LLM judge.
        
        Args:
            llm_client: LLM client for making judgment calls
            criteria: Custom evaluation criteria (uses defaults if None)
            judge_model: Specific model to use for judgments (optional)
        """
        self.llm_client = llm_client
        self.criteria = criteria or DEFAULT_CRITERIA
        self.judge_model = judge_model
        
        # Validate criteria weights
        total_weight = sum(c.weight for c in self.criteria)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Criteria weights sum to {total_weight}, not 1.0. Normalizing...")
            for criterion in self.criteria:
                criterion.weight /= total_weight
    
    async def judge_result(
        self, 
        result_data: dict[str, Any],
        prompt: str
    ) -> JudgmentResult:
        """Judge a single evaluation result.
        
        Args:
            result_data: Complete result data from smolval evaluation
            prompt: Original task prompt
            
        Returns:
            JudgmentResult with scores and analysis
        """
        result = result_data["result"]
        scores = []
        
        for criterion in self.criteria:
            try:
                score = await self._evaluate_criterion(criterion, result_data, prompt)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to evaluate criterion {criterion.name}: {e}")
                # Fallback score
                scores.append(JudgmentScore(
                    criterion=criterion.name,
                    score=0.0,
                    reasoning=f"Evaluation failed: {e}",
                ))
        
        # Calculate weighted overall score
        overall_score = sum(
            score.score * criterion.weight 
            for score, criterion in zip(scores, self.criteria, strict=True)
        )
        
        # Generate summary
        summary = await self._generate_summary(scores, result_data, prompt)
        
        # Extract strengths, weaknesses, suggestions
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []
        
        for score in scores:
            if score.details:
                if "strengths" in score.details:
                    all_strengths.extend(score.details["strengths"])
                if "weaknesses" in score.details:
                    all_weaknesses.extend(score.details["weaknesses"])
                if "suggestions" in score.details:
                    all_suggestions.extend(score.details["suggestions"])
        
        return JudgmentResult(
            overall_score=overall_score,
            scores=scores,
            summary=summary,
            strengths=all_strengths[:5],  # Top 5
            weaknesses=all_weaknesses[:5],  # Top 5  
            suggestions=all_suggestions[:5]  # Top 5
        )
    
    async def _evaluate_criterion(
        self, 
        criterion: JudgmentCriteria,
        result_data: dict[str, Any],
        prompt: str
    ) -> JudgmentScore:
        """Evaluate a single criterion."""
        result = result_data["result"]
        
        # Prepare context data for the criterion
        context = self._prepare_context(criterion.name, result_data, prompt)
        
        # Format the judgment prompt
        judgment_prompt = criterion.prompt_template.format(**context)
        
        # Make the judgment call
        messages = [
            LLMMessage(role="system", content="You are an expert evaluator of AI agent performance. Provide thorough, objective assessments based on the given criteria."),
            LLMMessage(role="user", content=judgment_prompt)
        ]
        
        response = await self.llm_client.generate(
            messages=messages,
            model=self.judge_model
        )
        
        # Parse the JSON response
        try:
            judgment_data = json.loads(response.content)
            
            return JudgmentScore(
                criterion=criterion.name,
                score=float(judgment_data["score"]),
                reasoning=judgment_data["reasoning"],
                details=judgment_data
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse judgment response for {criterion.name}: {e}")
            logger.debug(f"Raw response: {response.content}")
            
            # Fallback - try to extract score from text
            content = response.content.lower()
            if "score" in content:
                # Try to find a decimal number after "score"
                import re
                match = re.search(r'score[":]*\s*([0-9]*\.?[0-9]+)', content)
                if match:
                    score = min(1.0, max(0.0, float(match.group(1))))
                else:
                    score = 0.5  # Default neutral score
            else:
                score = 0.5
            
            return JudgmentScore(
                criterion=criterion.name,
                score=score,
                reasoning=f"Could not parse structured response: {response.content}",
                details={"parse_error": str(e)},
            )
    
    def _prepare_context(
        self, 
        criterion_name: str,
        result_data: dict[str, Any],
        prompt: str
    ) -> dict[str, Any]:
        """Prepare context data for a specific criterion evaluation."""
        result = result_data["result"]
        
        context = {
            "prompt": prompt,
            "final_answer": result["final_answer"],
            "success": result["success"],
            "total_iterations": result["total_iterations"],
            "execution_time_seconds": result["execution_time_seconds"],
            "failed_tool_calls": result.get("failed_tool_calls", 0)
        }
        
        if criterion_name == "reasoning_quality":
            # Prepare step-by-step reasoning summary
            steps = result.get("steps", [])
            reasoning_steps = []
            for i, step in enumerate(steps, 1):
                step_summary = f"Step {i} (Iteration {step.get('iteration', i)}):\n"
                step_summary += f"  Thought: {step.get('thought', 'N/A')}\n"
                if step.get('action'):
                    step_summary += f"  Action: {step['action']}\n"
                    if step.get('action_input'):
                        step_summary += f"  Input: {json.dumps(step['action_input'], indent=4)}\n"
                if step.get('observation'):
                    obs = step['observation']
                    if len(obs) > 500:
                        obs = obs[:500] + "... (truncated)"
                    step_summary += f"  Observation: {obs}\n"
                reasoning_steps.append(step_summary)
            
            context["reasoning_steps"] = "\n".join(reasoning_steps)
        
        elif criterion_name == "process_efficiency":
            # Prepare tool usage summary
            steps = result.get("steps", [])
            tool_usage = {}
            for step in steps:
                if step.get('action'):
                    tool_name = step['action']
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
            
            usage_summary = []
            for tool, count in tool_usage.items():
                usage_summary.append(f"- {tool}: {count} times")
            
            context["tool_usage_summary"] = "\n".join(usage_summary) if usage_summary else "No tools used"
        
        elif criterion_name == "task_understanding":
            # Prepare approach summary
            steps = result.get("steps", [])
            if steps:
                first_thoughts = [step.get('thought', '') for step in steps[:3]]
                context["approach_summary"] = " → ".join(first_thoughts)
            else:
                context["approach_summary"] = "No reasoning steps recorded"
        
        return context
    
    async def _generate_summary(
        self, 
        scores: list[JudgmentScore],
        result_data: dict[str, Any],
        prompt: str
    ) -> str:
        """Generate an overall summary of the judgment."""
        result = result_data["result"]
        
        # Create summary prompt
        score_summaries = []
        for score in scores:
            score_summaries.append(f"- {score.criterion}: {score.score:.2f} - {score.reasoning[:100]}...")
        
        summary_prompt = f"""
Based on the following individual criterion evaluations, provide a brief overall summary (2-3 sentences) of this agent's performance:

Task: {prompt[:200]}...
Success: {result['success']}

Criterion Scores:
{chr(10).join(score_summaries)}

Overall Assessment:
"""
        
        messages = [
            LLMMessage(role="system", content="Provide a concise, balanced summary of the agent's performance."),
            LLMMessage(role="user", content=summary_prompt)
        ]
        
        try:
            response = await self.llm_client.generate(
                messages=messages,
                model=self.judge_model
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            avg_score = sum(s.score for s in scores) / len(scores)
            return f"Agent achieved an average score of {avg_score:.2f} across all criteria. Performance was {'strong' if avg_score > 0.7 else 'moderate' if avg_score > 0.4 else 'weak'}."


class JudgedResult(BaseModel):
    """Result with LLM judgment included."""
    
    original_result: dict[str, Any]
    judgment: JudgmentResult
    metadata: dict[str, Any] | None = None