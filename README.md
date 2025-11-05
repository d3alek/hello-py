hello-py
===

## Task: Shape-Preserving Smoothing Resample

This task asks the model to implement a data preprocessing function commonly used in ML engineering:

**`smoothing_resample(arr: list[float], target_len: int) -> list[float]`**: Deterministically resamples an array to a target length while preserving overall shape (peaks, valleys, monotonic segments) and signal energy.

### Why This Task?

This function is essential for data preprocessing in ML pipelines, where arrays need to be resampled to match expected dimensions while preserving the character of the signal. The challenge lies in understanding that **downsampling requires area-weighted averaging** rather than simple point-sampling or interpolation.

### The RL Objective: Teaching Area-Preserving Resampling

Through iterative refinement, this task guides LLMs away from a common pitfall and toward proper signal processing:

**The Core Problem**: When models use **simple linear interpolation** for both upsampling and downsampling (the most obvious approach), they fail to preserve signal energy and shape. Linear interpolation creates a smooth curve between points but loses critical information about the distribution of values between samples.

**The Solution Path**: Models need to discover **area-preserving techniques** for downsampling:
- **Upsampling**: Linear interpolation is appropriate (creates smooth transitions)
- **Downsampling**: Must use area-weighted averaging within windows to preserve energy/mean
- **Key Insight**: Each output point should aggregate values from a window of input points, not just interpolate between endpoints

### Test Suite (4 Essential Checks)

1. **Basic Functionality**: Tests length correctness, identity case (`target_len == len(arr)` returns exact copy), and constant array handling
2. **Determinism**: Same inputs must produce bitwise-identical outputs
3. **Shape Preservation**: Tests that peaks stay at similar relative positions and variance doesn't artificially increase (catches global scaling)
   - Test case: `[1.0, 2.0, 10.0, 3.0, 2.0]` → 3 elements
   - Checks: `mean_ratio < 0.2`, `peak_loc_diff < 0.4`, variance preserved
4. **Anti-Scaling Test** (Most Discriminative): Ensures valleys are preserved without global scaling
   - Test case: `[1.0, 5.0, 1.0, 1.0, 1.0, 10.0, 1.0]` → 4 elements  
   - Checks: `mean_ratio < 0.2`, `min(output) < 2.5` (valley preserved near ~1.0)
   - This catches solutions that artificially scale all values to match a target mean

### The Hint Strategy

The prompt includes a critical hint:
> "Hint: When down-sampling, consider how to preserve the 'energy' or 'area' of the signal. Simple point-sampling or interpolation can lose important information. Think about averaging or aggregating values within windows."

This guides models toward the correct approach without prescribing a specific algorithm. The task rewards understanding over memorization.

### Success Criteria

**Target: 10-40% success rate** (30% observed with hint)

**Why this is challenging**:
- Linear interpolation is intuitive and works for upsampling, but models must recognize it's insufficient for downsampling
- The anti-scaling test requires understanding that windowed averaging should use **sequential non-overlapping windows** for best results
- Models must balance competing requirements: smoothness, energy preservation, and exact identity for the no-op case

**Successful solutions** implement:
- Separate logic paths for upsampling vs downsampling
- Area-weighted averaging with proper window overlap calculation
- Sequential window partitioning (e.g., `window_i = [i * len(arr) / target_len, (i+1) * len(arr) / target_len]`)

### Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

   By default, the script runs 10 iterations sequentially. To enable verbose logging, modify `main()` call:
   ```python
   asyncio.run(main(verbose=True))
   ```
