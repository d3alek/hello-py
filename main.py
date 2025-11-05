import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 4000
MAX_STEPS = 20

logger = logging.getLogger(__name__)


class GradingResult(TypedDict):
    success: bool
    checks_passed: list[str]
    checks_failed: list[str]


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """Tool that evaluates Python expressions using exec with persistent namespace."""
    if not hasattr(python_expression_tool, "namespace"):
        python_expression_tool.namespace = {}
    
    try:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, python_expression_tool.namespace, python_expression_tool.namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Tool for submitting the final answer."""
    return {"answer": answer, "submitted": True}


def calc_mean_ratio(arr1: list[float], arr2: list[float]) -> float:
    """Calculate relative mean difference between two arrays."""
    mean1 = sum(arr1) / len(arr1) if arr1 else 0
    mean2 = sum(arr2) / len(arr2) if arr2 else 0
    return abs((mean2 - mean1) / mean1) if mean1 != 0 else 0


def calc_variance(arr: list[float]) -> float:
    """Calculate variance of array."""
    if not arr:
        return 0
    mean = sum(arr) / len(arr)
    return sum((x - mean)**2 for x in arr) / len(arr)


def calc_peak_position_ratio(arr: list[float]) -> float:
    """Calculate relative position of peak (0.0 = start, 1.0 = end)."""
    if len(arr) <= 1:
        return 0
    peak_idx = arr.index(max(arr))
    return peak_idx / (len(arr) - 1)


def create_prompt(max_steps: int = MAX_STEPS) -> str:
    """Create the prompt for the smoothing resample task."""
    return f"""You are implementing a Python function for data preprocessing that is commonly used in ML engineering.

Task: Implement `smoothing_resample(arr: list[float], target_len: int) -> list[float]` with the following specifications:

- Deterministically returns a list of length `target_len` preserving overall shape
- Must be deterministic: same inputs always produce same outputs (bitwise-equal)
- Must handle up-sampling (target_len > len(arr)) and down-sampling (target_len < len(arr))
- Must preserve the overall shape: peaks, valleys, and monotonic segments should be maintained
- Must handle degenerate input (constant array) exactly and consistently
- Must return floats (not strings) with reasonable numeric precision
- For identity case (target_len == len(arr)), must return values equal (element-wise) to input (exact match)

Requirements:
- Must handle edge cases correctly (constant arrays, empty arrays, single element)
- Use the python_expression tool to write and test your implementation
- Your last step must be to submit your complete code using submit_answer (submit the full code containing the function definition as a string)

Hint: When down-sampling, consider how to preserve the "energy" or "area" of the signal. Simple point-sampling or interpolation can lose important information. Think about averaging or aggregating values within windows."""


def grade_implementation(code: str, namespace: dict) -> GradingResult:
    """Grade the implementation with 4 essential tests."""
    checks_passed = []
    checks_failed = []
    TOL = 1e-9
    
    func = namespace.get("smoothing_resample")
    
    if not func or not callable(func):
        checks_failed.append("smoothing_resample function not found")
        return GradingResult(success=False, checks_passed=checks_passed, checks_failed=checks_failed)
    
    try:
        arr = [1.0, 2.0, 3.0]
        if (len(func(arr, 5)) == 5 and 
            func(arr, 3) == arr and
            all(abs(x - 3.0) < TOL for x in func([3.0, 3.0, 3.0], 5))):
            checks_passed.append("Basic: length, identity, constant")
        else:
            checks_failed.append("Basic: functionality failed")
    except Exception as e:
        checks_failed.append(f"Basic: Exception - {str(e)}")
    
    try:
        arr = [1.0, 2.0, 3.0]
        if func(arr, 7) == func(arr, 7):
            checks_passed.append("Determinism: bitwise reproducible")
        else:
            checks_failed.append("Determinism: not reproducible")
    except Exception as e:
        checks_failed.append(f"Determinism: Exception - {str(e)}")
    
    try:
        peak_arr = [1.0, 2.0, 10.0, 3.0, 2.0]
        downsampled = func(peak_arr, 3)
        mean_ratio = calc_mean_ratio(peak_arr, downsampled)
        peak_loc_diff = abs(calc_peak_position_ratio(peak_arr) - calc_peak_position_ratio(downsampled))
        orig_var = calc_variance(peak_arr)
        down_var = calc_variance(downsampled)
        var_ratio = abs((down_var - orig_var) / orig_var) if orig_var != 0 else 0
        var_increased = down_var > orig_var * 1.1
        is_likely_scaling = var_increased or (mean_ratio < 0.01 and var_ratio > 0.50 and down_var > orig_var)
        if mean_ratio < 0.2 and peak_loc_diff < 0.4 and (var_ratio < 0.50 or down_var <= orig_var) and not is_likely_scaling:
            checks_passed.append("Shape: peak location and variance preserved")
        else:
            checks_failed.append(f"Shape: failed - mean_ratio={mean_ratio:.3f}, peak_loc_diff={peak_loc_diff:.3f}, var_ratio={var_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Shape: Exception - {str(e)}")
    
    try:
        bimodal_arr = [1.0, 5.0, 1.0, 1.0, 1.0, 10.0, 1.0]
        bimodal_down = func(bimodal_arr, 4)
        mean_ratio = calc_mean_ratio(bimodal_arr, bimodal_down)
        if len(bimodal_down) >= 2:
            valley_preserved = min(bimodal_down) < 2.5
            if mean_ratio < 0.2 and valley_preserved:
                checks_passed.append("Anti-scaling: valleys preserved (no global scaling)")
            else:
                checks_failed.append(f"Anti-scaling: failed - mean_ratio={mean_ratio:.3f}, valley={min(bimodal_down):.3f} (expected ~1.0)")
        else:
            checks_failed.append("Anti-scaling: insufficient output length")
    except Exception as e:
        checks_failed.append(f"Anti-scaling: Exception - {str(e)}")
    
    success = len(checks_failed) == 0
    return GradingResult(success=success, checks_passed=checks_passed, checks_failed=checks_failed)


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = MAX_STEPS,
    model: str = "claude-haiku-4-5",
    verbose: bool = False,
    run_id: int | None = None,
    num_runs: int | None = None,
) -> str | None:
    """Runs an agent loop with the given prompt and tools. Returns submitted code or code from namespace."""
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    all_code = []

    for step in range(max_steps):
        if run_id is not None and num_runs is not None:
            step_msg = f"\n=== Run {run_id}/{num_runs} Step {step + 1}/{max_steps} ==="
        elif run_id is not None:
            step_msg = f"\n=== Run {run_id} Step {step + 1}/{max_steps} ==="
        else:
            step_msg = f"\n=== Step {step + 1}/{max_steps} ==="
        
        logger.info(step_msg)

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"]
        if response.stop_reason == "max_tokens":
            msg = f"Model reached max_tokens limit {MAX_TOKENS}. Increase MAX_TOKENS, simplify your task, or update the code."
            logger.warning(msg)

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    logger.debug(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        logger.debug(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        assert isinstance(tool_input, dict) and "expression" in tool_input
                        expression = tool_input["expression"]
                        all_code.append(expression)
                        if verbose:
                            logger.debug(f"\nInput:\n```\n{expression}\n```")
                        result = handler(expression)
                        if verbose:
                            logger.debug(f"\nOutput:\n```\n{result}\n```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    logger.info(f"Agent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                logger.info("No tool use in response, ending loop.")
            break

    if verbose:
        logger.warning(f"Reached maximum steps ({max_steps}) without submitting answer.")
    
    if all_code:
        return "\n".join(all_code)
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = False,
) -> tuple[int, bool, GradingResult]:
    """Run a single test iteration."""
    if verbose:
        run_logs_dir = Path("run_logs")
        run_logs_dir.mkdir(exist_ok=True)
        run_dir = run_logs_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = f"run_{run_id}_{timestamp}.txt"
        output_path = run_dir / output_file
        
        file_handler = logging.FileHandler(output_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RUN {run_id}/{num_runs}")
    logger.info(f"{'=' * 60}")

    python_expression_tool.namespace = {}

    code = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=MAX_STEPS,
        verbose=verbose,
        run_id=run_id,
        num_runs=num_runs,
    )

    if code:
        namespace = python_expression_tool.namespace.copy()
        grading_result = grade_implementation(code, namespace)
        success = grading_result["success"]
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{'✓' if success else '✗'} Run {run_id}: {'SUCCESS' if success else 'FAILURE'}")
        
        logger.info(f"\nPassed Checks ({len(grading_result['checks_passed'])}):")
        for check in grading_result['checks_passed']:
            logger.info(f"  ✓ {check}")
        
        if grading_result['checks_failed']:
            logger.info(f"\nFailed Checks ({len(grading_result['checks_failed'])}):")
            for check in grading_result['checks_failed']:
                logger.info(f"  ✗ {check}")
        
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status} Run {run_id}: {len(grading_result['checks_passed'])}/{len(grading_result['checks_passed']) + len(grading_result['checks_failed'])} checks passed")
    else:
        raise RuntimeError(f"Run {run_id}: No code collected from agent")

    logger.info(f"{'=' * 60}\n")
    
    if verbose:
        logger.removeHandler(file_handler)
        file_handler.close()

    return run_id, success, grading_result


async def main(verbose: bool = True):
    """Main function to run the RL task."""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression. Use print() to output something. Returns stdout. Functions defined in previous calls are available in later calls.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    prompt = create_prompt(max_steps=MAX_STEPS)
    num_runs = 10
    model = "claude-haiku-4-5"

    logger.info(f"Running {num_runs} test iterations sequentially...")
    logger.info(f"Model: {model}")
    logger.info("=" * 60)

    results = []
    for i in range(num_runs):
        result = await run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            verbose=verbose,
        )
        results.append(result)

    successes = sum(success for _, success, _ in results)
    pass_rate = (successes / num_runs) * 100
    
    logger.info(f"\n{'=' * 60}")
    logger.info("Final Results:")
    logger.info(f"  Passed: {successes}/{num_runs}")
    logger.info(f"  Failed: {num_runs - successes}/{num_runs}")
    logger.info(f"  Pass Rate: {pass_rate:.1f}%")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
