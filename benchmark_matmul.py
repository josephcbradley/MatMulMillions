#!/usr/bin/env python3
"""
Benchmark script for large matrix multiplication and eigendecomposition.

This script benchmarks:
1. Matrix multiplication with a large matrix X (100000x1000)
2. Eigendecomposition of C = X'*X (1000x1000)

Uses pyperf for robust benchmarking with proper warmup and statistics.
"""

import numpy as np
import pyperf


# Global variables to avoid recreation overhead in benchmark loops
X = None
C = None


def setup_matrices():
    """Setup matrices once before benchmarking."""
    global X, C
    n_rows = 100000
    n_cols = 1000
    
    X = np.random.randn(n_rows, n_cols)
    
    C = X.T @ X


def benchmark_matmul(loops):
    """Benchmark function for matrix multiplication."""
    t0 = pyperf.perf_counter()
    for _ in range(loops):
        X.T @ X
    return pyperf.perf_counter() - t0


def benchmark_eigendecomposition(loops):
    """Benchmark function for eigendecomposition."""
    t0 = pyperf.perf_counter()
    for _ in range(loops):
        np.linalg.eigh(C)
    return pyperf.perf_counter() - t0


def main():
    """Main benchmarking routine using pyperf."""
    print("=" * 70)
    print("Matrix Multiplication and Eigendecomposition Benchmark (pyperf)")
    print("=" * 70)
    print()
    
    # Setup matrices once
    setup_matrices()
    
    # Create runner
    runner = pyperf.Runner()
    
    # Benchmark 1: Matrix multiplication X'*X
    print("-" * 70)
    print("BENCHMARK 1: Matrix Multiplication (X'*X)")
    print(f"Matrix X shape: {X.shape}")
    print("-" * 70)
    runner.bench_time_func('matmul', benchmark_matmul)
    
    # Benchmark 2: Eigendecomposition of C
    print()
    print("-" * 70)
    print("BENCHMARK 2: Eigendecomposition of C")
    print(f"Matrix C shape: {C.shape}")
    print("-" * 70)
    runner.bench_time_func('eigendecomposition', benchmark_eigendecomposition)


if __name__ == "__main__":
    main()
