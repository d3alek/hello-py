"""
RL Task: Shape-preserving smoothing resample + linear rescale utilities
Implements the first candidate task from task-candidate.txt.

This task tests the model's ability to implement numeric preprocessing functions
with proper edge case handling, determinism, and shape preservation.
"""

import asyncio
import json
import math
import os
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MAX_TOKENS = 8000
MAX_STEPS = 5

# Rate limiting: limit concurrent API calls to avoid rate limit errors
# Adjust based on your rate limits (e.g., 10k output tokens/min for Haiku)
MAX_CONCURRENT_REQUESTS = 3  # Conservative default
REQUEST_DELAY = 0.5  # Seconds to wait between starting requests


class GradingResult(TypedDict):
    """Result of grading the submitted code."""
    success: bool
    checks_passed: list[str]
    checks_failed: list[str]
    details: dict[str, Any]


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Maintains persistent namespace across calls so functions defined in previous steps
    are available in later steps.
    """
    # Persistent namespace shared across all calls
    if not hasattr(python_expression_tool, "namespace"):
        python_expression_tool.namespace = {}
    
    if not hasattr(python_expression_tool, "all_code"):
        python_expression_tool.all_code = []
    
    try:
        python_expression_tool.all_code.append(expression)
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, python_expression_tool.namespace, python_expression_tool.namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def get_collected_code() -> str:
    """Get all collected code from python_expression tool calls."""
    if hasattr(python_expression_tool, "all_code"):
        return "\n".join(python_expression_tool.all_code)
    return ""


def get_namespace() -> dict:
    """Get the current namespace from python_expression tool."""
    if hasattr(python_expression_tool, "namespace"):
        return python_expression_tool.namespace.copy()
    return {}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Tool for submitting the final answer."""
    return {"answer": answer, "submitted": True}


def create_prompt() -> str:
    """Create the prompt for the smoothing resample + linear rescale task."""
    return """You are implementing two Python functions for data preprocessing that are commonly used in ML engineering.

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
- Use the python_expression tool to write your implementation
- Test your functions with the provided test cases to ensure they work correctly
- Once you're confident, submit your answer using submit_answer

Important constraints:
- Do NOT use random sampling or any non-deterministic operations
- Do NOT return strings instead of floats
- For constant arrays, the output must preserve that constant (exact equality)
- For identity resampling (target_len == len(arr)), output must exactly match input

You can use any valid algorithm (linear interpolation, anti-alias smoothing + rebinning, Lanczos, piecewise-aggregate approximation, windowed averaging, etc.) as long as it satisfies all the requirements above, especially mean preservation."""


def create_test_harness() -> str:
    """Create a test harness that the model can use to test their implementation."""
    return """# Test harness for smoothing_resample and linear_rescale
import math

def test_smoothing_resample():
    '''Test harness for smoothing_resample function.'''
    tests = []
    
    # Test 1: Identity case (target_len == len(arr))
    arr1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result1 = smoothing_resample(arr1, 5)
    tests.append(("Identity case", result1 == arr1, f"Expected {arr1}, got {result1}"))
    
    # Test 2: Constant array
    arr2 = [3.0, 3.0, 3.0]
    result2 = smoothing_resample(arr2, 5)
    tests.append(("Constant array upsample", all(abs(x - 3.0) < 1e-9 for x in result2), 
                 f"Expected all 3.0, got {result2}"))
    
    # Test 3: Downsample
    arr3 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result3 = smoothing_resample(arr3, 3)
    tests.append(("Downsample length", len(result3) == 3, f"Expected length 3, got {len(result3)}"))
    
    # Test 4: Upsample
    arr4 = [1.0, 5.0, 2.0]
    result4 = smoothing_resample(arr4, 10)
    tests.append(("Upsample length", len(result4) == 10, f"Expected length 10, got {len(result4)}"))
    
    # Test 5: Determinism
    arr5 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result5a = smoothing_resample(arr5, 7)
    result5b = smoothing_resample(arr5, 7)
    tests.append(("Determinism", result5a == result5b, "Two calls with same input produced different results"))
    
    return tests

def test_linear_rescale():
    '''Test harness for linear_rescale function.'''
    tests = []
    
    # Test 1: Basic rescaling
    arr1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result1 = linear_rescale(arr1, 0.0, 10.0)
    tests.append(("Rescale min correct", abs(min(result1) - 0.0) < 1e-9, 
                 f"Expected min=0.0, got min={min(result1)}"))
    tests.append(("Rescale max correct", abs(max(result1) - 10.0) < 1e-9,
                 f"Expected max=10.0, got max={max(result1)}"))
    
    # Test 2: Constant array
    arr2 = [5.0, 5.0, 5.0]
    result2 = linear_rescale(arr2, 0.0, 10.0)
    # For constant array, all values should be equal (any constant is acceptable)
    first_val = result2[0]
    all_equal = all(abs(x - first_val) < 1e-9 for x in result2)
    tests.append(("Constant array rescale", all_equal,
                 f"Expected all values equal, got {result2}"))
    
    # Test 3: Monotonic ordering preserved
    arr3 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result3 = linear_rescale(arr3, 0.0, 10.0)
    is_monotonic = all(result3[i] <= result3[i+1] for i in range(len(result3)-1))
    tests.append(("Monotonic ordering", is_monotonic, f"Ordering not preserved: {result3}"))
    
    # Test 4: Determinism
    arr4 = [1.0, 2.0, 3.0]
    result4a = linear_rescale(arr4, 0.0, 1.0)
    result4b = linear_rescale(arr4, 0.0, 1.0)
    tests.append(("Determinism", result4a == result4b, "Two calls with same input produced different results"))
    
    return tests

# Run tests
print("Testing smoothing_resample...")
smoothing_tests = test_smoothing_resample()
for name, passed, msg in smoothing_tests:
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {name} - {msg}")

print()
print("Testing linear_rescale...")
rescale_tests = test_linear_rescale()
for name, passed, msg in rescale_tests:
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {name} - {msg}")
"""


def extract_functions(namespace: dict) -> tuple[Callable | None, Callable | None]:
    """
    Extract smoothing_resample and linear_rescale functions from namespace.
    Returns (smoothing_resample_func, linear_rescale_func) or (None, None) if not found.
    """
    smoothing_resample_func = namespace.get("smoothing_resample")
    linear_rescale_func = namespace.get("linear_rescale")
    
    # Verify they are callable
    if smoothing_resample_func and not callable(smoothing_resample_func):
        smoothing_resample_func = None
    if linear_rescale_func and not callable(linear_rescale_func):
        linear_rescale_func = None
    
    return smoothing_resample_func, linear_rescale_func


def grade_implementation(code: str, namespace: dict) -> GradingResult:
    """
    Grade the implementation against all 6 grading contract items.
    
    Grading Contract (from task-candidate.txt):
    1. Output length equals target_len for smoothing_resample
    2. Identity case: target_len == len(arr) returns exact match
    3. Constant input: both functions return arrays where all values equal that constant
    4. linear_rescale maps min->new_min and max->new_max (within tolerance), preserves monotonic ordering
    5. Determinism: multiple runs with same inputs produce bitwise-equal outputs
    6. Shape preservation: resampled outputs have mean and peak location within bounds
    
    Returns GradingResult with success status and detailed check results.
    """
    checks_passed = []
    checks_failed = []
    details = {}
    
    # Extract functions
    smoothing_resample_func, linear_rescale_func = extract_functions(namespace)
    
    if not smoothing_resample_func:
        checks_failed.append("smoothing_resample function not found")
        return GradingResult(
            success=False,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details={"error": "smoothing_resample function not found in code"}
        )
    
    if not linear_rescale_func:
        checks_failed.append("linear_rescale function not found")
        return GradingResult(
            success=False,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details={"error": "linear_rescale function not found in code"}
        )
    
    # Tolerance for floating point comparisons
    TOL = 1e-9
    
    # Check 1: Output length equals target_len for smoothing_resample
    try:
        test_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        for target_len in [3, 5, 10]:
            result = smoothing_resample_func(test_arr, target_len)
            if not isinstance(result, list):
                checks_failed.append(f"Check 1: smoothing_resample returned {type(result)}, not list")
                break
            if len(result) != target_len:
                checks_failed.append(f"Check 1: Expected length {target_len}, got {len(result)}")
                break
        else:
            checks_passed.append("Check 1: Output length equals target_len")
    except Exception as e:
        checks_failed.append(f"Check 1: Exception - {str(e)}")
    
    # Check 2: Identity case - exact match
    try:
        test_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = smoothing_resample_func(test_arr, len(test_arr))
        if result == test_arr:
            checks_passed.append("Check 2: Identity case returns exact match")
        else:
            checks_failed.append(f"Check 2: Identity case failed - expected {test_arr}, got {result}")
            details["identity_case"] = {"expected": test_arr, "got": result}
    except Exception as e:
        checks_failed.append(f"Check 2: Exception - {str(e)}")
    
    # Check 3: Constant input - both functions preserve constant
    try:
        # Test smoothing_resample with constant
        const_arr = [3.0, 3.0, 3.0]
        resample_result = smoothing_resample_func(const_arr, 5)
        if all(abs(x - 3.0) < TOL for x in resample_result):
            checks_passed.append("Check 3a: smoothing_resample preserves constant")
        else:
            checks_failed.append(f"Check 3a: Expected all 3.0, got {resample_result}")
        
        # Test linear_rescale with constant
        # For constant array, min==max, so linear mapping is undefined
        # The spec says "all values equal that constant (or mapped constant for linear_rescale)"
        # We accept any constant value (new_min, new_max, or midpoint are all reasonable)
        rescale_result = linear_rescale_func(const_arr, 0.0, 10.0)
        # Check that all values are the same (constant) using tolerance
        if len(rescale_result) > 0:
            first_val = rescale_result[0]
            all_equal = all(abs(x - first_val) < TOL for x in rescale_result)
            if all_equal:
                checks_passed.append("Check 3b: linear_rescale preserves constant (all values equal)")
            else:
                checks_failed.append(f"Check 3b: Expected all values to be equal, got {rescale_result}")
        else:
            checks_failed.append("Check 3b: linear_rescale returned empty result")
    except Exception as e:
        checks_failed.append(f"Check 3: Exception - {str(e)}")
    
    # Check 4: linear_rescale maps extremes correctly and preserves monotonic ordering
    try:
        test_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        new_min, new_max = 0.0, 10.0
        result = linear_rescale_func(test_arr, new_min, new_max)
        
        # Check min mapping
        min_mapped = min(result)
        max_mapped = max(result)
        min_ok = abs(min_mapped - new_min) < TOL
        max_ok = abs(max_mapped - new_max) < TOL
        
        if min_ok and max_ok:
            checks_passed.append("Check 4a: linear_rescale maps min->new_min and max->new_max")
        else:
            checks_failed.append(f"Check 4a: min mapping failed (got {min_mapped}, expected {new_min}) or "
                               f"max mapping failed (got {max_mapped}, expected {new_max})")
        
        # Check monotonic ordering
        is_monotonic = all(result[i] <= result[i+1] for i in range(len(result)-1))
        if is_monotonic:
            checks_passed.append("Check 4b: linear_rescale preserves monotonic ordering")
        else:
            checks_failed.append(f"Check 4b: Monotonic ordering not preserved: {result}")
        
        # Also test with descending input
        desc_arr = [5.0, 4.0, 3.0, 2.0, 1.0]
        desc_result = linear_rescale_func(desc_arr, 0.0, 10.0)
        is_desc_monotonic = all(desc_result[i] >= desc_result[i+1] for i in range(len(desc_result)-1))
        if is_desc_monotonic:
            checks_passed.append("Check 4c: linear_rescale preserves descending monotonic ordering")
        else:
            checks_failed.append(f"Check 4c: Descending monotonic ordering not preserved")
    except Exception as e:
        checks_failed.append(f"Check 4: Exception - {str(e)}")
    
    # Check 5: Determinism
    try:
        # Test smoothing_resample determinism
        test_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result1 = smoothing_resample_func(test_arr, 7)
        result2 = smoothing_resample_func(test_arr, 7)
        if result1 == result2:
            checks_passed.append("Check 5a: smoothing_resample is deterministic")
        else:
            checks_failed.append(f"Check 5a: smoothing_resample not deterministic - results differ")
        
        # Test linear_rescale determinism
        result3 = linear_rescale_func(test_arr, 0.0, 10.0)
        result4 = linear_rescale_func(test_arr, 0.0, 10.0)
        if result3 == result4:
            checks_passed.append("Check 5b: linear_rescale is deterministic")
        else:
            checks_failed.append(f"Check 5b: linear_rescale not deterministic - results differ")
    except Exception as e:
        checks_failed.append(f"Check 5: Exception - {str(e)}")
    
    # Check 6: Shape preservation (mean and peak location within bounds)
    try:
        # Test with array that has a clear peak
        peak_arr = [1.0, 2.0, 10.0, 3.0, 2.0]
        original_mean = sum(peak_arr) / len(peak_arr)
        original_peak_idx = peak_arr.index(max(peak_arr))
        
        # Upsample
        upsampled = smoothing_resample_func(peak_arr, 15)
        upsampled_mean = sum(upsampled) / len(upsampled)
        upsampled_peak_idx = upsampled.index(max(upsampled))
        
        # Mean should be approximately preserved (within 20% tolerance)
        mean_ratio = abs(upsampled_mean - original_mean) / original_mean if original_mean != 0 else abs(upsampled_mean)
        mean_ok = mean_ratio < 0.2
        
        # Peak location should be approximately preserved (within 30% of array length)
        # Normalize peak location to [0, 1]
        orig_peak_norm = original_peak_idx / (len(peak_arr) - 1) if len(peak_arr) > 1 else 0
        upsampled_peak_norm = upsampled_peak_idx / (len(upsampled) - 1) if len(upsampled) > 1 else 0
        peak_loc_diff = abs(upsampled_peak_norm - orig_peak_norm)
        peak_ok = peak_loc_diff < 0.3
        
        if mean_ok and peak_ok:
            checks_passed.append("Check 6: Shape preservation (mean and peak location)")
        else:
            checks_failed.append(f"Check 6: Shape preservation failed - mean_ratio={mean_ratio:.3f}, "
                               f"peak_loc_diff={peak_loc_diff:.3f}")
            details["shape_preservation"] = {
                "original_mean": original_mean,
                "upsampled_mean": upsampled_mean,
                "original_peak_idx": original_peak_idx,
                "upsampled_peak_idx": upsampled_peak_idx
            }
        
        # Also test downsampling
        downsampled = smoothing_resample_func(peak_arr, 3)
        downsampled_mean = sum(downsampled) / len(downsampled)
        downsampled_peak_idx = downsampled.index(max(downsampled))
        
        mean_ratio_down = abs(downsampled_mean - original_mean) / original_mean if original_mean != 0 else abs(downsampled_mean)
        orig_peak_norm = original_peak_idx / (len(peak_arr) - 1) if len(peak_arr) > 1 else 0
        downsampled_peak_norm = downsampled_peak_idx / (len(downsampled) - 1) if len(downsampled) > 1 else 0
        peak_loc_diff_down = abs(downsampled_peak_norm - orig_peak_norm)
        
        if mean_ratio_down < 0.2 and peak_loc_diff_down < 0.3:
            checks_passed.append("Check 6b: Shape preservation (downsampling)")
        else:
            checks_failed.append(f"Check 6b: Downsampling shape preservation failed")
        
        # Additional test: Different peak pattern (peak at start)
        peak_start_arr = [10.0, 1.0, 2.0, 3.0, 4.0]
        peak_start_mean = sum(peak_start_arr) / len(peak_start_arr)
        peak_start_down = smoothing_resample_func(peak_start_arr, 3)
        peak_start_down_mean = sum(peak_start_down) / len(peak_start_down)
        peak_start_ratio = abs(peak_start_down_mean - peak_start_mean) / peak_start_mean if peak_start_mean != 0 else abs(peak_start_down_mean)
        if peak_start_ratio < 0.2:
            checks_passed.append("Check 6c: Shape preservation (peak at start)")
        else:
            checks_failed.append(f"Check 6c: Peak at start shape preservation failed - mean_ratio={peak_start_ratio:.3f}")
        
        # Additional test: Isolated peak pattern
        isolated_peak_arr = [1.0, 1.0, 10.0, 1.0, 1.0]
        isolated_mean = sum(isolated_peak_arr) / len(isolated_peak_arr)
        isolated_down = smoothing_resample_func(isolated_peak_arr, 3)
        isolated_down_mean = sum(isolated_down) / len(isolated_down)
        isolated_ratio = abs(isolated_down_mean - isolated_mean) / isolated_mean if isolated_mean != 0 else abs(isolated_down_mean)
        if isolated_ratio < 0.2:
            checks_passed.append("Check 6d: Shape preservation (isolated peak)")
        else:
            checks_failed.append(f"Check 6d: Isolated peak shape preservation failed - mean_ratio={isolated_ratio:.3f}")
    except Exception as e:
        checks_failed.append(f"Check 6: Exception - {str(e)}")
    
    # Check 7: linear_rescale with reversed min/max (should reverse order and map correctly)
    try:
        test_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Reverse order: new_min > new_max
        result = linear_rescale_func(test_arr, 10.0, 0.0)
        
        # Should map min(arr) to new_min (10.0) and max(arr) to new_max (0.0)
        # Note: when reversed, the first element (min) maps to new_min, last (max) to new_max
        first_val = result[0]  # Should be min(arr) -> new_min
        last_val = result[-1]  # Should be max(arr) -> new_max
        min_ok = abs(first_val - 10.0) < TOL
        max_ok = abs(last_val - 0.0) < TOL
        
        # Should preserve descending monotonicity
        is_desc_monotonic = all(result[i] >= result[i+1] for i in range(len(result)-1))
        
        if min_ok and max_ok and is_desc_monotonic:
            checks_passed.append("Check 7: linear_rescale with reversed min/max")
        else:
            checks_failed.append(f"Check 7: Reversed min/max failed - first={first_val:.6f}, last={last_val:.6f}, desc_monotonic={is_desc_monotonic}")
    except Exception as e:
        checks_failed.append(f"Check 7: Exception - {str(e)}")
    
    # Check 8: Single element array handling
    try:
        # Test smoothing_resample with single element
        single_arr = [5.0]
        single_result = smoothing_resample_func(single_arr, 3)
        # Should preserve the constant value
        if all(abs(x - 5.0) < TOL for x in single_result):
            checks_passed.append("Check 8a: Single element array preserves constant")
        else:
            checks_failed.append(f"Check 8a: Single element not preserved - got {single_result}")
        
        # Test linear_rescale with single element
        single_rescale = linear_rescale_func(single_arr, 0.0, 10.0)
        # For constant array, should map to midpoint
        if all(abs(x - 5.0) < TOL for x in single_rescale):
            checks_passed.append("Check 8b: Single element linear_rescale")
        else:
            checks_failed.append(f"Check 8b: Single element rescale failed - got {single_rescale}")
    except Exception as e:
        checks_failed.append(f"Check 8: Exception - {str(e)}")
    
    # Overall success: all checks must pass
    success = len(checks_failed) == 0
    
    return GradingResult(
        success=success,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        details=details
    )


# Global semaphore for rate limiting API calls
_api_semaphore: asyncio.Semaphore | None = None

# Global progress tracking for Docker-style display
_progress_lines: dict[int, dict[str, Any]] = {}
_progress_lock: asyncio.Lock | None = None


def get_api_semaphore() -> asyncio.Semaphore:
    """Get or create the global API semaphore for rate limiting."""
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _api_semaphore


def get_progress_lock() -> asyncio.Lock:
    """Get or create the global progress lock."""
    global _progress_lock
    if _progress_lock is None:
        _progress_lock = asyncio.Lock()
    return _progress_lock


def update_progress(run_id: int, step: int | None = None, status: str = "running", 
                   checks_passed: int = 0, checks_failed: int = 0, total_checks: int = 0):
    """Update progress for a specific run."""
    _progress_lines[run_id] = {
        "step": step,
        "status": status,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "total_checks": total_checks,
    }


def print_progress(num_runs: int, clear_first: bool = False):
    """Print all progress lines in Docker-style format."""
    import sys
    
    if clear_first:
        # Number of lines to clear: header (1) + separator (1) + runs (num_runs) = num_runs + 2
        lines_to_clear = num_runs + 2
        # Move cursor up
        print(f"\033[{lines_to_clear}A", end="")
        # Clear each line
        for _ in range(lines_to_clear):
            print("\033[K", end="")  # Clear from cursor to end of line
        # Move cursor back to start of line
        print("\r", end="")
    
    # Print header
    print(f"{'Run':<6} {'Step':<6} {'Status':<12} {'Checks':<20}")
    print("-" * 50)
    
    # Print each run's status
    for run_id in range(1, num_runs + 1):
        progress = _progress_lines.get(run_id, {})
        step = progress.get("step", "-")
        status = progress.get("status", "waiting")
        checks_passed = progress.get("checks_passed", 0)
        checks_failed = progress.get("checks_failed", 0)
        total_checks = progress.get("total_checks", 0)
        
        # Format status
        if status == "finished":
            status_icon = "✓" if checks_failed == 0 else "✗"
            status_text = f"{status_icon} {status}"
        elif status == "running":
            status_text = f"→ {status}"
        else:
            status_text = f"○ {status}"
        
        # Format checks
        if total_checks > 0:
            checks_text = f"{checks_passed}/{total_checks} passed"
            if checks_failed > 0:
                checks_text += f" ({checks_failed} failed)"
        else:
            checks_text = "-"
        
        print(f"{run_id:<6} {str(step):<6} {status_text:<12} {checks_text:<20}")
    
    # Flush to ensure immediate display
    sys.stdout.flush()


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
    """
    Runs an agent loop with the given prompt and tools.
    Returns the collected code if successful, None otherwise.
    Uses a semaphore to limit concurrent API calls and avoid rate limits.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    semaphore = get_api_semaphore()

    for step in range(max_steps):
        # Update progress if tracking
        if run_id is not None:
            update_progress(run_id, step=step + 1, status="running")
            if num_runs is not None:
                async with get_progress_lock():
                    print_progress(num_runs, clear_first=True)
        # Acquire semaphore to limit concurrent API calls
        async with semaphore:
            # Add small delay between requests to avoid rate limits
            # This helps spread out requests over time
            if REQUEST_DELAY > 0:
                await asyncio.sleep(REQUEST_DELAY)
            
            response = await client.messages.create(
                model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
            )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                pass  # No verbose output
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        result = handler(tool_input["expression"])
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
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

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return collected code
            if submitted_answer is not None:
                return get_collected_code()
        else:
            # No tool use, conversation might be complete
            break
    
    # Return collected code even if no explicit submission
    return get_collected_code()


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    output_file: str | None = None,
) -> tuple[int, bool, GradingResult]:
    """Run a single test iteration."""
    # Create run_logs directory structure
    run_logs_dir = Path("run_logs")
    run_logs_dir.mkdir(exist_ok=True)
    
    # Create directory for this specific run
    run_dir = run_logs_dir / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Create output file for this run
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = f"first_candidate_run_{run_id}_{timestamp}.txt"
    
    output_path = run_dir / output_file
    
    def log(msg: str = ""):
        """Write to file only (progress is shown separately via print_progress)"""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 60}")
    log(f"RUN {run_id}/{num_runs}")
    log(f"{'=' * 60}")

    # Reset tool state for this run
    python_expression_tool.namespace = {}
    python_expression_tool.all_code = []

    # Initialize progress tracking
    update_progress(run_id, step=0, status="starting")
    
    code = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=MAX_STEPS,
        verbose=False,  # No verbose output to stdout
        run_id=run_id,
        num_runs=num_runs,
    )

    if code:
        # Get the namespace with executed functions
        namespace = get_namespace()
        
        log(f"\nCollected Code ({len(code)} chars):")
        log("-" * 60)
        log(code)
        log("-" * 60)

        # Grade the implementation
        grading_result = grade_implementation(code, namespace)
        
        success = grading_result["success"]
        total_checks = len(grading_result['checks_passed']) + len(grading_result['checks_failed'])
        
        # Update progress with final results
        update_progress(
            run_id,
            step=MAX_STEPS,
            status="finished",
            checks_passed=len(grading_result['checks_passed']),
            checks_failed=len(grading_result['checks_failed']),
            total_checks=total_checks,
        )
        
        log(f"\n{'=' * 60}")
        if success:
            log(f"✓ Run {run_id}: SUCCESS")
            log(f"  Passed checks: {len(grading_result['checks_passed'])}")
        else:
            log(f"✗ Run {run_id}: FAILURE")
            log(f"  Passed checks: {len(grading_result['checks_passed'])}")
            log(f"  Failed checks: {len(grading_result['checks_failed'])}")
        
        log(f"\nPassed Checks ({len(grading_result['checks_passed'])}):")
        for check in grading_result['checks_passed']:
            log(f"  ✓ {check}")
        
        if grading_result['checks_failed']:
            log(f"\nFailed Checks ({len(grading_result['checks_failed'])}):")
            for check in grading_result['checks_failed']:
                log(f"  ✗ {check}")
        
        if grading_result['details']:
            log(f"\nDetails:")
            for key, value in grading_result['details'].items():
                log(f"  {key}: {value}")
    else:
        success = False
        grading_result = GradingResult(
            success=False,
            checks_passed=[],
            checks_failed=["No code collected from agent"],
            details={"error": "No code was collected"}
        )
        # Update progress for failure case
        update_progress(
            run_id,
            step=MAX_STEPS,
            status="finished",
            checks_passed=0,
            checks_failed=1,
            total_checks=1,
        )
        log(f"✗ Run {run_id}: FAILURE - No code collected")

    log(f"{'=' * 60}\n")
    log(f"Output saved to: {output_path}")

    return run_id, success, grading_result


async def main(
    concurrent: bool = False,
    num_runs: int = 10,
    model: str = "claude-haiku-4-5",
    max_concurrent: int | None = None,
    request_delay: float | None = None,
):
    """Main function to run the RL task.
    
    Args:
        concurrent: Whether to run tests concurrently
        num_runs: Number of test iterations
        model: Anthropic model to use
        max_concurrent: Maximum concurrent API requests (overrides global default)
        request_delay: Delay between requests in seconds (overrides global default)
    """
    # Override global settings if provided
    global MAX_CONCURRENT_REQUESTS, REQUEST_DELAY, _api_semaphore
    if max_concurrent is not None:
        MAX_CONCURRENT_REQUESTS = max_concurrent
        _api_semaphore = None  # Reset so it gets recreated with new value
    if request_delay is not None:
        REQUEST_DELAY = request_delay
    
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression. Use print() to output something. Returns stdout. "
                          "Functions defined in previous calls are available in later calls.",
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

    # Create prompt
    full_prompt = create_prompt()

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print(f"Model: {model}")
    print(f"Max concurrent API requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Request delay: {REQUEST_DELAY}s")
    print(f"Output files will be saved in: run_logs/run_<id>/first_candidate_run_<id>_<timestamp>.txt")
    print("=" * 60)
    print()
    
    # Initialize progress tracking
    global _progress_lines
    _progress_lines = {}
    for i in range(1, num_runs + 1):
        update_progress(i, step=0, status="waiting")
    
    # Print initial progress display
    print_progress(num_runs, clear_first=False)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=full_prompt,
            tools=tools,
            tool_handlers=tool_handlers,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Update progress after each completion
            async with get_progress_lock():
                print_progress(num_runs, clear_first=True)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)
            # Update progress after each completion
            async with get_progress_lock():
                print_progress(num_runs, clear_first=True)
    
    # Final progress update
    print()  # New line after progress
    async with get_progress_lock():
        print_progress(num_runs, clear_first=False)

    # Sort by run_id
    results.sort(key=lambda x: x[0])

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate pass rate
    pass_rate = (successes / num_runs) * 100
    
    # Only show final summary (progress display already shows individual results)
    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
    
    # Check if pass rate is in target range
    if pass_rate < 10:
        print("\n⚠ Success rate is below 10% - task may be too difficult")
    elif pass_rate > 40:
        print("\n⚠ Success rate is above 40% - task may be too easy")
    else:
        print(f"\n✓ Success rate ({pass_rate:.1f}%) is in the target range (10-40%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RL Task: Shape-preserving smoothing resample + linear rescale utilities"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of test iterations to run (default: 10)"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run test iterations concurrently (faster but uses more API quota)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Anthropic model to use (default: claude-haiku-4-5)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=None,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(
        concurrent=args.concurrent,
        num_runs=args.num_runs,
        model=args.model,
        max_concurrent=args.max_concurrent,
        request_delay=args.request_delay,
    ))

