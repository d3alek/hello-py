import asyncio
import json
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 8000
MAX_STEPS = 20


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


def create_prompt(max_steps: int = MAX_STEPS) -> str:
    """Create the prompt for the smoothing resample + linear rescale task."""
    return f"""You are implementing two Python functions for data preprocessing that are commonly used in ML engineering.

Task: Implement two functions with the following specifications:

1. `smoothing_resample(arr: list[float], target_len: int) -> list[float]`
   - Deterministically returns a list of length `target_len` preserving overall shape
   - Must be deterministic: same inputs always produce same outputs (bitwise-equal)
   - Must handle up-sampling (target_len > len(arr)) and down-sampling (target_len < len(arr))
   - Must preserve monotonic segments and relative peaks as much as possible
   - Must preserve the mean of the array: when resampling, the mean of the output should be within 20% of the mean of the input (important for shape preservation)
   - Must handle degenerate input (constant array) exactly and consistently
   - Must return floats (not strings) with reasonable numeric precision
   - For identity case (target_len == len(arr)), must return values equal (element-wise) to input (exact match)

2. `linear_rescale(arr: list[float], new_min: float, new_max: float) -> list[float]`
   - Linearly maps arr's min->new_min and max->new_max
   - Must preserve relative ordering (if a_i < a_j then rescaled_i <= rescaled_j)
   - Must map min(arr) to new_min and max(arr) to new_max (within floating tolerance ~1e-9)
   - For constant input (all values equal), must return array where all values equal the mapped constant (exact equality)

Requirements:
- Both functions must be deterministic (no randomness)
- Both functions must handle edge cases correctly (constant arrays, empty arrays, single element)
- Output must be a list of floats, not strings
- For smoothing_resample: the mean preservation requirement is critical - when downsampling or upsampling, the output array's mean must be within 20% of the input array's mean
- Use the python_expression tool to write and test your implementation
- Your last step must be to submit your complete code using submit_answer (submit the full code containing both function definitions as a string)

You can use any valid algorithm (linear interpolation, anti-alias smoothing + rebinning, Lanczos, piecewise-aggregate approximation, windowed averaging, etc.) as long as it satisfies all the requirements above, especially mean preservation."""


def grade_implementation(code: str, namespace: dict) -> GradingResult:
    """Grade the implementation - streamlined checks."""
    checks_passed = []
    checks_failed = []
    TOL = 1e-9
    
    smoothing_resample_func = namespace.get("smoothing_resample")
    linear_rescale_func = namespace.get("linear_rescale")
    
    if not smoothing_resample_func or not callable(smoothing_resample_func):
        checks_failed.append("smoothing_resample function not found")
        return GradingResult(success=False, checks_passed=checks_passed, checks_failed=checks_failed)
    
    if not linear_rescale_func or not callable(linear_rescale_func):
        checks_failed.append("linear_rescale function not found")
        return GradingResult(success=False, checks_passed=checks_passed, checks_failed=checks_failed)
    
    # Basic checks (always pass - consolidated)
    try:
        arr = [1.0, 2.0, 3.0]
        if (len(smoothing_resample_func(arr, 5)) == 5 and 
            smoothing_resample_func(arr, 3) == arr and
            all(abs(x - 3.0) < TOL for x in smoothing_resample_func([3.0, 3.0, 3.0], 5))):
            checks_passed.append("Check 1-3: Basic functionality")
        else:
            checks_failed.append("Check 1-3: Basic functionality failed")
    except Exception as e:
        checks_failed.append(f"Check 1-3: Exception - {str(e)}")
    
    try:
        result = linear_rescale_func([1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 10.0)
        rescale_const = linear_rescale_func([3.0, 3.0, 3.0], 0.0, 10.0)
        if (abs(min(result) - 0.0) < TOL and abs(max(result) - 10.0) < TOL and
            len(rescale_const) > 0 and all(abs(x - rescale_const[0]) < TOL for x in rescale_const)):
            checks_passed.append("Check 4: linear_rescale basic")
        else:
            checks_failed.append("Check 4: linear_rescale basic failed")
    except Exception as e:
        checks_failed.append(f"Check 4: Exception - {str(e)}")
    
    try:
        arr = [1.0, 2.0, 3.0]
        if (smoothing_resample_func(arr, 7) == smoothing_resample_func(arr, 7) and
            linear_rescale_func(arr, 0.0, 10.0) == linear_rescale_func(arr, 0.0, 10.0)):
            checks_passed.append("Check 5: Determinism")
        else:
            checks_failed.append("Check 5: Determinism failed")
    except Exception as e:
        checks_failed.append(f"Check 5: Exception - {str(e)}")
    
    # Shape preservation checks (sometimes fail - keep all)
    try:
        peak_arr = [1.0, 2.0, 10.0, 3.0, 2.0]
        original_mean = sum(peak_arr) / len(peak_arr)
        downsampled = smoothing_resample_func(peak_arr, 3)
        mean_ratio = abs((sum(downsampled) / len(downsampled) - original_mean) / original_mean) if original_mean != 0 else 0
        if mean_ratio < 0.2:
            checks_passed.append("Check 6b: Downsampling shape preservation")
        else:
            checks_failed.append(f"Check 6b: Downsampling failed - mean_ratio={mean_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6b: Exception - {str(e)}")
    
    try:
        peak_start_arr = [10.0, 1.0, 2.0, 3.0, 4.0]
        peak_start_mean = sum(peak_start_arr) / len(peak_start_arr)
        peak_start_down = smoothing_resample_func(peak_start_arr, 3)
        mean_ratio = abs((sum(peak_start_down) / len(peak_start_down) - peak_start_mean) / peak_start_mean) if peak_start_mean != 0 else 0
        if mean_ratio < 0.2:
            checks_passed.append("Check 6c: Peak at start shape preservation")
        else:
            checks_failed.append(f"Check 6c: Peak at start failed - mean_ratio={mean_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6c: Exception - {str(e)}")
    
    try:
        isolated_peak_arr = [1.0, 1.0, 10.0, 1.0, 1.0]
        isolated_mean = sum(isolated_peak_arr) / len(isolated_peak_arr)
        isolated_down = smoothing_resample_func(isolated_peak_arr, 3)
        mean_ratio = abs((sum(isolated_down) / len(isolated_down) - isolated_mean) / isolated_mean) if isolated_mean != 0 else 0
        if mean_ratio < 0.2:
            checks_passed.append("Check 6d: Isolated peak shape preservation")
        else:
            checks_failed.append(f"Check 6d: Isolated peak failed - mean_ratio={mean_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6d: Exception - {str(e)}")
    
    try:
        single_rescale = linear_rescale_func([5.0], 0.0, 10.0)
        if not all(abs(x - 5.0) < TOL for x in single_rescale):
            checks_failed.append(f"Check 8b: Single element failed - got {single_rescale}")
        else:
            checks_passed.append("Check 8b: Single element linear_rescale")
    except Exception as e:
        checks_failed.append(f"Check 8b: Exception - {str(e)}")
    
    success = len(checks_failed) == 0
    return GradingResult(success=success, checks_passed=checks_passed, checks_failed=checks_failed)


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = MAX_STEPS,
    model: str = "claude-haiku-4-5",
    verbose: bool = False,
) -> str | None:
    """Runs an agent loop with the given prompt and tools. Returns submitted code or code from namespace."""
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    all_code = []

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"]
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        assert isinstance(tool_input, dict) and "expression" in tool_input
                        expression = tool_input["expression"]
                        all_code.append(expression)
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in expression.split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(expression)
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
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
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    # Return collected code from namespace if no submission
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
    def log(msg: str = ""):
        """Log to terminal if verbose is True."""
        if verbose:
            print(msg)

    log(f"\n{'=' * 60}")
    log(f"RUN {run_id}/{num_runs}")
    log(f"{'=' * 60}")

    # Reset tool state for this run
    python_expression_tool.namespace = {}

    code = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=MAX_STEPS,
        verbose=verbose,
    )

    if code:
        log(f"\nCollected Code ({len(code)} chars):")
        log("-" * 60)
        log(code)
        log("-" * 60)

        # Use namespace from tool for grading
        namespace = python_expression_tool.namespace.copy()
        grading_result = grade_implementation(code, namespace)
        success = grading_result["success"]
        
        log(f"\n{'=' * 60}")
        if success:
            log(f"✓ Run {run_id}: SUCCESS")
        else:
            log(f"✗ Run {run_id}: FAILURE")
        
        log(f"\nPassed Checks ({len(grading_result['checks_passed'])}):")
        for check in grading_result['checks_passed']:
            log(f"  ✓ {check}")
        
        if grading_result['checks_failed']:
            log(f"\nFailed Checks ({len(grading_result['checks_failed'])}):")
            for check in grading_result['checks_failed']:
                log(f"  ✗ {check}")
    else:
        success = False
        grading_result = GradingResult(
            success=False,
            checks_passed=[],
            checks_failed=["No code collected from agent"]
        )
        log(f"✗ Run {run_id}: FAILURE - No code collected")

    log(f"{'=' * 60}\n")

    return run_id, success, grading_result


async def main(verbose: bool = False):
    """Main function to run the RL task."""
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

    print(f"Running {num_runs} test iterations sequentially...")
    print(f"Model: {model}")
    print("=" * 60)

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
    
    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
