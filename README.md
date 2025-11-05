hello-py
===

## Task: Shape-Preserving Smoothing Resample

This task asks the model to implement a data preprocessing function commonly used in ML engineering:

**`smoothing_resample(arr: list[float], target_len: int) -> list[float]`**: Deterministically resamples an array to a target length while preserving overall shape, monotonicity, and most critically, the mean of the array (within 20% tolerance).

### Why This Task?

This function is essential for data preprocessing in ML pipelines, where arrays need to be resampled to match expected dimensions while preserving statistical properties. The challenge lies in maintaining the mean during downsampling, which requires understanding area-preserving resampling techniques rather than simple interpolation.

### General Model Problem Discovered

Through extensive testing, we discovered a critical weakness in how LLMs handle **mean preservation during downsampling** in the `smoothing_resample` function:

**The Problem**: When models use simple linear interpolation (the most obvious approach), they fail to preserve the mean of the input array when downsampling, especially with arrays containing sharp peaks. Linear interpolation creates a smooth curve between points, but doesn't account for the statistical property of mean preservation.

**Evidence from Failure Modes**:
- **Check 6b, 6c, 6d**: These checks test mean preservation during downsampling with different peak patterns (center peak, peak at start, isolated peak). Models consistently fail these with mean ratios of 0.33-0.43 (well above the 0.20 threshold), indicating they're losing ~60-70% of the mean information during downsampling.
- **Check 6b** (Downsampling with center peak): `[1.0, 2.0, 10.0, 3.0, 2.0]` → mean_ratio often ~0.33
- **Check 6c** (Peak at start): `[10.0, 1.0, 2.0, 3.0, 4.0]` → mean_ratio often ~0.33  
- **Check 6d** (Isolated peak): `[1.0, 1.0, 10.0, 1.0, 1.0]` → mean_ratio often ~0.43

**Why This Matters**: Simple linear interpolation doesn't preserve area-under-the-curve (which is proportional to mean for discrete arrays). To properly preserve the mean, models need to use techniques like:
- Area-preserving resampling (integrating and rebinning)
- Anti-aliasing with proper windowing
- Piecewise-aggregate approximation with proper averaging
- Weighted averaging that accounts for the distribution

The task targets a **10-40% success rate** because while basic linear interpolation is easy to implement, achieving mean preservation requires understanding the statistical implications of resampling operations, which most models miss.

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
