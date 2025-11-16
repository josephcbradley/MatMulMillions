# Matrix Multiplication and Eigendecomposition Benchmarks

This repository contains robust benchmark scripts for evaluating the performance of large matrix operations in both Python (NumPy) and Julia.

## Benchmarks Performed

Both scripts benchmark the following operations on a large matrix **X** (100,000 × 1,000):

1. **Matrix Multiplication**: `C = X' * X` (resulting in a 1,000 × 1,000 matrix)
2. **Eigendecomposition**: Computing eigenvalues and eigenvectors of the matrix `C`

## Python Implementation

### Requirements

- Python 3.x
- NumPy
- pyperf (Python Performance Benchmark Suite)

### Installation

```bash
uv add numpy pyperf
```

### Usage

Run the benchmark with default settings:
```bash
uv run benchmark_matmul.py
```

Run with fast mode (fewer samples for quick testing):
```bash
uv run benchmark_matmul.py --fast
```

Run with rigorous mode (more samples for accurate results):
```bash
uv run benchmark_matmul.py --rigorous
```

Run with quiet mode (suppress warnings):
```bash
uv run benchmark_matmul.py --quiet
```

### Key Features

- Uses `pyperf` for professional-grade benchmarking
- Automatic calibration of loop counts
- Multiple processes and runs for statistical significance
- Warmup iterations to account for JIT/caching effects
- Detailed statistics including mean, standard deviation, min, max

### Example Output

```
matmul: Mean +- std dev: 1.25 sec +- 0.17 sec
eigendecomposition: Mean +- std dev: 233 ms +- 72 ms
```

## Julia Implementation

### Requirements

- Julia 1.x
- LinearAlgebra (standard library)
- BenchmarkTools

### Installation

First install Julia, then add the required package:
```bash
sudo snap install julia
```

In Julia REPL:
```julia
using Pkg
Pkg.add("BenchmarkTools")
```

### Usage

Run the benchmark:
```bash
julia benchmark_matmul.jl
```

### Key Features

- Uses `BenchmarkTools.@benchmark` for robust benchmarking
- Automatic warmup and multiple sample collection
- Advanced statistics including median, mean, memory allocations
- Detection and filtering of garbage collection effects
- Time distribution analysis

### Example Output

```
BenchmarkTools.Trial:
  memory estimate:  7.63 MiB
  allocs estimate:  2
  minimum time:     723.456 ms (0.00% GC)
  median time:      745.123 ms (0.00% GC)
  mean time:        748.890 ms (0.00% GC)
  maximum time:     780.234 ms (0.00% GC)
```

## Comparison

Both implementations provide:
- Professional-grade benchmarking with proper warmup
- Multiple samples for statistical reliability
- Detection of unstable benchmarks
- Detailed timing statistics

### Python (pyperf) Advantages:
- Cross-platform process isolation
- Built-in system tuning recommendations
- JSON export for result comparison
- Integration with pyperf ecosystem

### Julia (BenchmarkTools) Advantages:
- Memory allocation tracking
- Garbage collection detection
- More detailed time distribution analysis
- Typically faster execution due to Julia's JIT compilation

## Performance Tips

### For Python:
- Run `python3 -m pyperf system tune` to optimize system for benchmarking
- Use `--rigorous` flag for production benchmarks
- Consider CPU affinity with `--affinity` flag
- Save results with `-o results.json` for comparison

### For Julia:
- First run will be slower due to JIT compilation
- Use `@btime` from BenchmarkTools for quick checks
- Consider using `BenchmarkTools.tune!()` for custom configurations
- Check for garbage collection overhead in results

## Matrix Sizes

The benchmarks are configured for:
- **X**: 100,000 rows × 1,000 columns (~763 MB)
- **C**: 1,000 × 1,000 (~8 MB)

These can be modified in the source files by changing `n_rows` and `n_cols` variables.

## Notes

- Results will vary based on hardware, CPU architecture, and BLAS implementation
- NumPy typically uses OpenBLAS or MKL for optimized matrix operations
- Julia uses OpenBLAS by default, but can be configured to use other BLAS libraries
- Both implementations benefit from multi-threaded BLAS operations
