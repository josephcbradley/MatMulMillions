#!/usr/bin/env julia
"""
Benchmark script for large matrix multiplication and eigendecomposition.

This script benchmarks:
1. Matrix multiplication with a large matrix X (100000x1000)
2. Eigendecomposition of C = X'*X (1000x1000)

Uses BenchmarkTools for robust benchmarking with proper warmup and statistics.
"""

using LinearAlgebra
using BenchmarkTools
using Printf


# Global variables to store matrices
X = Matrix{Float64}(undef, 0, 0)
C = Matrix{Float64}(undef, 0, 0)


"""
    setup_matrices!()

Setup matrices once before benchmarking.
"""
function setup_matrices!(X, C)
    global X, C
    n_rows = 100000
    n_cols = 1000
    
    X .= randn(n_rows, n_cols)
    
    mul!(C, X', X)
end


"""
    benchmark_matmul()

Benchmark function for matrix multiplication.
"""
function benchmark_matmul()
    return X' * X
end


"""
    benchmark_eigendecomposition()

Benchmark function for eigendecomposition.
"""
function benchmark_eigendecomposition()
    return eigen(Symmetric(C))
end


"""
    main()

Main benchmarking routine using BenchmarkTools.
"""
function main()
    println("="^70)
    println("Matrix Multiplication and Eigendecomposition Benchmark (BenchmarkTools)")
    println("="^70)
    println()
    
    # Setup matrices once
    setup_matrices!()
    
    # Benchmark 1: Matrix multiplication X'*X
    println("-"^70)
    println("BENCHMARK 1: Matrix Multiplication (X'*X)")
    println("Matrix X shape: $(size(X))")
    println("-"^70)
    println("Running benchmark with warmup and multiple samples...")
    matmul_bench = @benchmark benchmark_matmul()
    println("\nMatrix Multiplication Results:")
    display(matmul_bench)
    println("\n")
    
    # Benchmark 2: Eigendecomposition of C
    println()
    println("-"^70)
    println("BENCHMARK 2: Eigendecomposition of C")
    println("Matrix C shape: $(size(C))")
    println("-"^70)
    println("Running benchmark with warmup and multiple samples...")
    eigen_bench = @benchmark benchmark_eigendecomposition()
    println("\nEigendecomposition Results:")
    display(eigen_bench)
    println("\n")
    
    # Summary
    println()
    println("="^70)
    println("SUMMARY")
    println("="^70)
    @printf("Matrix multiplication median time:  %.4f seconds\n", median(matmul_bench).time / 1e9)
    @printf("Matrix multiplication mean time:    %.4f seconds\n", mean(matmul_bench).time / 1e9)
    @printf("Eigendecomposition median time:     %.4f seconds\n", median(eigen_bench).time / 1e9)
    @printf("Eigendecomposition mean time:       %.4f seconds\n", mean(eigen_bench).time / 1e9)
    println("="^70)
end

main()