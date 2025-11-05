import asyncio
import json
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
    """Create the prompt for the smoothing resample task."""
    return f"""You are implementing a Python function for data preprocessing that is commonly used in ML engineering.

Task: Implement `smoothing_resample(arr: list[float], target_len: int) -> list[float]` with the following specifications:

- Deterministically returns a list of length `target_len` preserving overall shape
- Must be deterministic: same inputs always produce same outputs (bitwise-equal)
- Must handle up-sampling (target_len > len(arr)) and down-sampling (target_len < len(arr))
- Must preserve monotonic segments and relative peaks as much as possible
- Must preserve the mean of the array: when resampling, the mean of the output should be within 20% of the mean of the input (important for shape preservation)
- Must handle degenerate input (constant array) exactly and consistently
- Must return floats (not strings) with reasonable numeric precision
- For identity case (target_len == len(arr)), must return values equal (element-wise) to input (exact match)

Requirements:
- Must handle edge cases correctly (constant arrays, empty arrays, single element)
- Use the python_expression tool to write and test your implementation
- Your last step must be to submit your complete code using submit_answer (submit the full code containing the function definition as a string)

You can use any valid algorithm (linear interpolation, anti-alias smoothing + rebinning, Lanczos, piecewise-aggregate approximation, windowed averaging, etc.) as long as it satisfies all the requirements above."""


def grade_implementation(code: str, namespace: dict) -> GradingResult:
    """Grade the implementation - streamlined checks."""
    checks_passed = []
    checks_failed = []
    TOL = 1e-9
    
    smoothing_resample_func = namespace.get("smoothing_resample")
    
    if not smoothing_resample_func or not callable(smoothing_resample_func):
        checks_failed.append("smoothing_resample function not found")
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
        arr = [1.0, 2.0, 3.0]
        if smoothing_resample_func(arr, 7) == smoothing_resample_func(arr, 7):
            checks_passed.append("Check 4: Determinism")
        else:
            checks_failed.append("Check 4: Determinism failed")
    except Exception as e:
        checks_failed.append(f"Check 4: Exception - {str(e)}")
    
    # Shape preservation checks (sometimes fail - keep all)
    try:
        peak_arr = [1.0, 2.0, 10.0, 3.0, 2.0]
        original_mean = sum(peak_arr) / len(peak_arr)
        original_peak_idx = peak_arr.index(max(peak_arr))
        downsampled = smoothing_resample_func(peak_arr, 3)
        downsampled_mean = sum(downsampled) / len(downsampled)
        mean_ratio = abs((downsampled_mean - original_mean) / original_mean) if original_mean != 0 else 0
        # Check peak location is preserved (not just scaled)
        downsampled_peak_idx = downsampled.index(max(downsampled))
        peak_pos_ratio = original_peak_idx / (len(peak_arr) - 1) if len(peak_arr) > 1 else 0
        downsampled_peak_pos_ratio = downsampled_peak_idx / (len(downsampled) - 1) if len(downsampled) > 1 else 0
        peak_loc_diff = abs(peak_pos_ratio - downsampled_peak_pos_ratio)
        # Check variance preservation
        # Note: Legitimate area-preserving resampling (averaging) can reduce variance naturally
        # Global scaling changes variance by scale_factor^2, which is different from averaging
        original_var = sum((x - original_mean)**2 for x in peak_arr) / len(peak_arr) if len(peak_arr) > 0 else 0
        downsampled_var = sum((x - downsampled_mean)**2 for x in downsampled) / len(downsampled) if len(downsampled) > 0 else 0
        var_ratio = abs((downsampled_var - original_var) / original_var) if original_var != 0 else 0
        # Detect global scaling: variance increases or changes proportionally (not just decreases from averaging)
        # Global scaling: if they scale by factor f, variance changes by f^2
        # Averaging: variance naturally decreases (values get combined)
        # Key insight: if variance INCREASES significantly, it's likely scaling (averaging only decreases)
        var_increased = downsampled_var > original_var * 1.1  # More than 10% increase
        # Also check if variance change matches scaling pattern: if mean is perfect but var changed a lot, suspicious
        is_likely_scaling = var_increased or (mean_ratio < 0.01 and var_ratio > 0.50 and downsampled_var > original_var)
        # Allow variance reduction (from averaging) up to 50%, but flag increases as suspicious
        if mean_ratio < 0.2 and peak_loc_diff < 0.4 and (var_ratio < 0.50 or downsampled_var <= original_var) and not is_likely_scaling:
            checks_passed.append("Check 6b: Downsampling shape preservation")
        else:
            checks_failed.append(f"Check 6b: Downsampling failed - mean_ratio={mean_ratio:.3f}, peak_loc_diff={peak_loc_diff:.3f}, var_ratio={var_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6b: Exception - {str(e)}")
    
    try:
        peak_start_arr = [10.0, 1.0, 2.0, 3.0, 4.0]
        peak_start_mean = sum(peak_start_arr) / len(peak_start_arr)
        peak_start_down = smoothing_resample_func(peak_start_arr, 3)
        mean_ratio = abs((sum(peak_start_down) / len(peak_start_down) - peak_start_mean) / peak_start_mean) if peak_start_mean != 0 else 0
        # Check that peak is at start (index 0)
        peak_at_start = peak_start_down.index(max(peak_start_down)) == 0
        if mean_ratio < 0.2 and peak_at_start:
            checks_passed.append("Check 6c: Peak at start shape preservation")
        else:
            checks_failed.append(f"Check 6c: Peak at start failed - mean_ratio={mean_ratio:.3f}, peak_at_start={peak_at_start}")
    except Exception as e:
        checks_failed.append(f"Check 6c: Exception - {str(e)}")
    
    try:
        isolated_peak_arr = [1.0, 1.0, 10.0, 1.0, 1.0]
        isolated_mean = sum(isolated_peak_arr) / len(isolated_peak_arr)
        isolated_down = smoothing_resample_func(isolated_peak_arr, 3)
        mean_ratio = abs((sum(isolated_down) / len(isolated_down) - isolated_mean) / isolated_mean) if isolated_mean != 0 else 0
        # Check that peak is still isolated (middle element should be max)
        isolated_peak_idx = isolated_peak_arr.index(max(isolated_peak_arr))
        isolated_down_peak_idx = isolated_down.index(max(isolated_down))
        peak_in_middle = isolated_down_peak_idx == len(isolated_down) // 2
        if mean_ratio < 0.2 and peak_in_middle:
            checks_passed.append("Check 6d: Isolated peak shape preservation")
        else:
            checks_failed.append(f"Check 6d: Isolated peak failed - mean_ratio={mean_ratio:.3f}, peak_in_middle={peak_in_middle}")
    except Exception as e:
        checks_failed.append(f"Check 6d: Exception - {str(e)}")
    
    try:
        # Test extreme downsampling - harder case that forces larger mean correction
        # This array has a large peak, so linear interpolation will lose mean significantly
        extreme_arr = [1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0]
        extreme_mean = sum(extreme_arr) / len(extreme_arr)
        extreme_down = smoothing_resample_func(extreme_arr, 2)
        extreme_down_mean = sum(extreme_down) / len(extreme_down)
        extreme_mean_ratio = abs((extreme_down_mean - extreme_mean) / extreme_mean) if extreme_mean != 0 else 0
        # Check variance - but note: averaging can legitimately reduce variance (especially in extreme downsampling)
        # Global scaling would INCREASE variance, while averaging DECREASES it
        extreme_original_var = sum((x - extreme_mean)**2 for x in extreme_arr) / len(extreme_arr) if len(extreme_arr) > 0 else 0
        extreme_down_var = sum((x - extreme_down_mean)**2 for x in extreme_down) / len(extreme_down) if len(extreme_down) > 0 else 0
        extreme_var_ratio = abs((extreme_down_var - extreme_original_var) / extreme_original_var) if extreme_original_var != 0 else 0
        # For extreme downsampling (9->2), averaging can collapse variance to near-zero (legitimate)
        # Global scaling would increase variance, which is the real problem
        extreme_var_increased = extreme_down_var > extreme_original_var * 1.1
        # Allow variance reduction (from averaging), but flag increases (from scaling)
        if extreme_mean_ratio < 0.2 and (extreme_var_ratio < 0.90 or not extreme_var_increased):
            checks_passed.append("Check 6e: Extreme downsampling (9->2)")
        else:
            checks_failed.append(f"Check 6e: Extreme downsampling failed - mean_ratio={extreme_mean_ratio:.3f}, var_ratio={extreme_var_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6e: Exception - {str(e)}")
    
    # NEW: Test for global scaling detection - asymmetric distribution
    try:
        # Array with low baseline and high peak - global scaling will distort the baseline
        # Expected: low values stay low, peak preserved
        # Global scaling: ALL values scaled up uniformly (distorts shape)
        asymmetric_arr = [0.5, 0.5, 0.5, 10.0, 0.5, 0.5]
        asymmetric_mean = sum(asymmetric_arr) / len(asymmetric_arr)
        asymmetric_down = smoothing_resample_func(asymmetric_arr, 3)
        asymmetric_down_mean = sum(asymmetric_down) / len(asymmetric_down)
        
        # Check mean preservation
        asym_mean_ratio = abs((asymmetric_down_mean - asymmetric_mean) / asymmetric_mean) if asymmetric_mean != 0 else 0
        
        # Key test: Check if baseline values are preserved
        # In proper resampling, low values should stay close to 0.5
        # With global scaling, they'll be scaled up significantly
        baseline_value = 0.5
        min_output = min(asymmetric_down)
        # For proper area-averaging: min should be close to baseline (within 50%)
        # For global scaling: min will be scaled up significantly (>2x)
        baseline_preserved = min_output < baseline_value * 1.8
        
        if asym_mean_ratio < 0.2 and baseline_preserved:
            checks_passed.append("Check 7a: Asymmetric distribution - no global scaling")
        else:
            checks_failed.append(f"Check 7a: Asymmetric failed - mean_ratio={asym_mean_ratio:.3f}, min_output={min_output:.3f} (expected ~{baseline_value})")
    except Exception as e:
        checks_failed.append(f"Check 7a: Exception - {str(e)}")
    
    # NEW: Test bimodal distribution with different peak heights
    try:
        # Two peaks of different heights - global scaling distorts relative heights
        bimodal_arr = [1.0, 5.0, 1.0, 1.0, 1.0, 10.0, 1.0]
        bimodal_mean = sum(bimodal_arr) / len(bimodal_arr)
        bimodal_down = smoothing_resample_func(bimodal_arr, 4)
        bimodal_down_mean = sum(bimodal_down) / len(bimodal_down)
        
        # Check mean
        bimodal_mean_ratio = abs((bimodal_down_mean - bimodal_mean) / bimodal_mean) if bimodal_mean != 0 else 0
        
        # Original has peaks at ~5 and ~10 (ratio 1:2)
        # Proper resampling should preserve this ratio roughly
        # Global scaling preserves ratio perfectly but distorts baseline
        sorted_output = sorted(bimodal_down, reverse=True)
        if len(sorted_output) >= 2:
            peak1 = sorted_output[0]
            peak2 = sorted_output[1]
            # Check that valleys (lowest values) stay close to baseline (1.0)
            valley = min(bimodal_down)
            valley_preserved = valley < 2.5  # Should be close to 1.0, not scaled up
            
            if bimodal_mean_ratio < 0.2 and valley_preserved:
                checks_passed.append("Check 7b: Bimodal distribution - shape preserved")
            else:
                checks_failed.append(f"Check 7b: Bimodal failed - mean_ratio={bimodal_mean_ratio:.3f}, valley={valley:.3f} (expected ~1.0)")
        else:
            checks_failed.append("Check 7b: Bimodal - insufficient output length")
    except Exception as e:
        checks_failed.append(f"Check 7b: Exception - {str(e)}")
    
    # NEW: Test upsampling with asymmetric distribution
    try:
        # Upsampling should also not use global scaling
        up_arr = [0.2, 0.2, 8.0, 0.2]
        up_mean = sum(up_arr) / len(up_arr)
        up_sampled = smoothing_resample_func(up_arr, 8)
        up_sampled_mean = sum(up_sampled) / len(up_sampled)
        
        # Check mean
        up_mean_ratio = abs((up_sampled_mean - up_mean) / up_mean) if up_mean != 0 else 0
        
        # Check baseline preservation - low values should stay low
        # Linear interpolation: baseline ~0.2, should not be scaled
        # Global scaling: would scale everything uniformly
        min_up = min(up_sampled)
        baseline_up_preserved = min_up < 0.4  # Should be close to 0.2
        
        if up_mean_ratio < 0.2 and baseline_up_preserved:
            checks_passed.append("Check 7c: Upsampling asymmetric - no scaling")
        else:
            checks_failed.append(f"Check 7c: Upsampling failed - mean_ratio={up_mean_ratio:.3f}, min={min_up:.3f} (expected ~0.2)")
    except Exception as e:
        checks_failed.append(f"Check 7c: Exception - {str(e)}")
    
    # NEW: Test step function - should preserve sharp transitions
    try:
        # Step function: low->high transition
        step_arr = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
        step_mean = sum(step_arr) / len(step_arr)
        step_down = smoothing_resample_func(step_arr, 3)
        step_down_mean = sum(step_down) / len(step_down)
        
        # Check mean
        step_mean_ratio = abs((step_down_mean - step_mean) / step_mean) if step_mean != 0 else 0
        
        # Expected: [~1, ~5, ~9] or similar - transition should be smooth but endpoints preserved
        # Global scaling: would scale everything uniformly, distorting the step
        # Key: Check that endpoints are close to original values
        first_val = step_down[0]
        last_val = step_down[-1]
        first_preserved = abs(first_val - 1.0) < 1.5  # Should be close to 1.0
        last_preserved = abs(last_val - 9.0) < 1.5   # Should be close to 9.0
        
        if step_mean_ratio < 0.2 and first_preserved and last_preserved:
            checks_passed.append("Check 7d: Step function - endpoints preserved")
        else:
            checks_failed.append(f"Check 7d: Step function failed - mean_ratio={step_mean_ratio:.3f}, first={first_val:.3f}, last={last_val:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 7d: Exception - {str(e)}")
    
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
    log_fn: Callable[[str], None] | None = None,
) -> str | None:
    """Runs an agent loop with the given prompt and tools. Returns submitted code or code from namespace."""
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    all_code = []

    for step in range(max_steps):
        step_msg = ""
        if run_id is not None and num_runs is not None:
            step_msg = f"\n=== Run {run_id}/{num_runs} Step {step + 1}/{max_steps} ==="
        elif run_id is not None:
            step_msg = f"\n=== Run {run_id} Step {step + 1}/{max_steps} ==="
        else:
            step_msg = f"\n=== Step {step + 1}/{max_steps} ==="
        
        print(step_msg)
        if log_fn:
            log_fn(step_msg)

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"]
        if response.stop_reason == "max_tokens":
            msg = (
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )
            print(msg)
            if log_fn:
                log_fn(msg)

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    msg = f"Assistant: {content.text}"
                    print(msg)
                    if log_fn:
                        log_fn(msg)
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        msg = f"Using tool: {tool_name}"
                        print(msg)
                        if log_fn:
                            log_fn(msg)

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
                            if log_fn:
                                log_fn("\nInput:")
                                log_fn("```")
                                for line in expression.split("\n"):
                                    log_fn(f"{line}")
                                log_fn("```")
                        result = handler(expression)
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                            if log_fn:
                                log_fn("\nOutput:")
                                log_fn("```")
                                log_fn(str(result))
                                log_fn("```")
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
                    msg = f"\nAgent submitted answer: {submitted_answer}"
                    print(msg)
                    if log_fn:
                        log_fn(msg)
                return submitted_answer
        else:
            if verbose:
                msg = "\nNo tool use in response, ending loop."
                print(msg)
                if log_fn:
                    log_fn(msg)
            break

    if verbose:
        msg = f"\nReached maximum steps ({max_steps}) without submitting answer."
        print(msg)
        if log_fn:
            log_fn(msg)
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
    # Create run_logs directory structure if verbose
    log_file = None
    if verbose:
        run_logs_dir = Path("run_logs")
        run_logs_dir.mkdir(exist_ok=True)
        
        run_dir = run_logs_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = f"first_candidate_run_{run_id}_{timestamp}.txt"
        output_path = run_dir / output_file
        log_file = open(output_path, "w", encoding="utf-8")
    
    def log(msg: str = ""):
        """Log to file if verbose, otherwise do nothing."""
        if verbose and log_file:
            log_file.write(msg + "\n")
            log_file.flush()

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
        run_id=run_id,
        num_runs=num_runs,
        log_fn=log if verbose else None,
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
        
        # Always print concise results
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} Run {run_id}: {len(grading_result['checks_passed'])}/{len(grading_result['checks_passed']) + len(grading_result['checks_failed'])} checks passed")
    else:
        raise RuntimeError(f"Run {run_id}: No code collected from agent - this indicates a developer error or API issue")

    log(f"{'=' * 60}\n")
    
    if log_file:
        log(f"Output saved to: {output_path}")
        log_file.close()

    return run_id, success, grading_result


async def main(verbose: bool = True):
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
