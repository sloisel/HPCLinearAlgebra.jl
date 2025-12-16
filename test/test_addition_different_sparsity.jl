using Test
using MPI
using LinearAlgebraMPI
using LinearAlgebraMPI: SparseMatrixMPI, VectorMPI, io0, clear_plan_cache!
using SparseArrays
using LinearAlgebra

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[TEST] Testing matrix addition with different sparsity patterns (nranks=$nranks)")

# This test reproduces a bug where adding matrices with different sparsity patterns
# can fail due to cached MatrixPlan having stale local_ranges

@testset "Addition with different sparsity patterns" begin
    # Create two matrices with different sparsity patterns
    # A has nonzeros in different positions than B

    n = 8

    # Matrix A: tridiagonal pattern
    A_native = spdiagm(n, n,
        -1 => ones(n-1),
        0 => 2*ones(n),
        1 => ones(n-1)
    )

    # Matrix B: different pattern (only diagonal and one off-diagonal)
    B_native = spdiagm(n, n,
        0 => 3*ones(n),
        2 => 0.5*ones(n-2)  # Different off-diagonal than A
    )

    A_mpi = SparseMatrixMPI{Float64}(A_native)
    B_mpi = SparseMatrixMPI{Float64}(B_native)

    println(io0(), "[TEST] A nnz: $(nnz(A_native)), B nnz: $(nnz(B_native))")

    # Test A + B
    println(io0(), "[TEST] Computing A + B...")
    C_mpi = A_mpi + B_mpi
    C_native = SparseMatrixCSC(C_mpi)
    C_expected = A_native + B_native

    @test norm(C_native - C_expected) < 1e-10
    println(io0(), "[TEST] A + B passed")

    # Now test with matrices produced from multiplication (like in f2)
    # This is closer to the actual failing case

    # Create D operators similar to fem1d
    dx = spdiagm(n, n, 0 => -ones(n), 1 => ones(n-1))
    dx[end, end] = 0  # Boundary
    id = spdiagm(n, n, 0 => ones(n))  # Identity matrix with Float64

    D_dx = SparseMatrixMPI{Float64}(dx)
    D_id = SparseMatrixMPI{Float64}(id)

    # Create a diagonal weight matrix
    w = ones(n) * 0.5
    W = SparseMatrixMPI{Float64}(spdiagm(n, n, 0 => w))

    println(io0(), "[TEST] Computing D' * W * D products...")

    # Compute products with different structure
    # M1 = id' * W * dx (structure of dx)
    # M2 = dx' * W * id (structure of dx')
    M1 = D_id' * W * D_dx
    M2 = D_dx' * W * D_id

    println(io0(), "[TEST] M1 nnz: $(nnz(SparseMatrixCSC(M1)))")
    println(io0(), "[TEST] M2 nnz: $(nnz(SparseMatrixCSC(M2)))")

    # This addition previously failed with BoundsError due to cached plan issue
    println(io0(), "[TEST] Computing M1 + M2 (this is where the bug occurred)...")
    M_sum = M1 + M2
    M_sum_native = SparseMatrixCSC(M_sum)

    # Compute expected result using native Julia
    M1_expected = id' * spdiagm(n, n, 0 => w) * dx
    M2_expected = dx' * spdiagm(n, n, 0 => w) * id
    M_sum_expected = M1_expected + M2_expected

    @test norm(M_sum_native - M_sum_expected) < 1e-10
    println(io0(), "[TEST] M1 + M2 passed")

    # Also test the accumulation pattern used in Hessian assembly
    println(io0(), "[TEST] Testing Hessian-style accumulation...")

    # Start with one product
    H = D_dx' * W * D_dx
    H_native_start = SparseMatrixCSC(H)

    # Add another product with different structure
    H = H + D_id' * W * D_id
    H_native_step2 = SparseMatrixCSC(H)

    # Add cross terms (this pattern caused the original bug)
    cross1 = D_dx' * W * D_id
    cross2 = D_id' * W * D_dx
    cross_sum = cross1 + cross2
    H = H + cross_sum
    H_native_final = SparseMatrixCSC(H)

    # Compute expected
    H_expected = dx' * spdiagm(n,n,0=>w) * dx +
                 id' * spdiagm(n,n,0=>w) * id +
                 dx' * spdiagm(n,n,0=>w) * id +
                 id' * spdiagm(n,n,0=>w) * dx

    @test norm(H_native_final - H_expected) < 1e-10
    println(io0(), "[TEST] Hessian-style accumulation passed")

    # Test the exact pattern from MultiGridBarrierMPI that triggered the bug
    # The bug occurred when:
    # 1. We compute foo * dx (structure A)
    # 2. We compute dx' * foo (structure B, different from A)
    # 3. We add A + B
    println(io0(), "[TEST] Testing exact bug-triggering pattern...")

    # Create fresh diagonal matrix each time (different values to ensure different matrices)
    foo1 = SparseMatrixMPI{Float64}(spdiagm(n, n, 0 => 0.3 * ones(n)))
    foo2 = SparseMatrixMPI{Float64}(spdiagm(n, n, 0 => 0.7 * ones(n)))

    # Compute products: these have DIFFERENT sparsity patterns
    # prod1 = foo1 * dx has structure of dx
    # prod2 = dx' * foo2 has structure of dx'
    prod1 = foo1 * D_dx
    prod2 = D_dx' * foo2

    prod1_native = SparseMatrixCSC(prod1)
    prod2_native = SparseMatrixCSC(prod2)

    println(io0(), "[TEST] prod1 (foo*dx) nnz: $(nnz(prod1_native))")
    println(io0(), "[TEST] prod2 (dx'*foo) nnz: $(nnz(prod2_native))")

    # This is where the bug occurred - adding matrices with different structure
    println(io0(), "[TEST] Adding prod1 + prod2...")
    sum_result = prod1 + prod2
    sum_native = SparseMatrixCSC(sum_result)

    # Expected
    foo1_native = spdiagm(n, n, 0 => 0.3 * ones(n))
    foo2_native = spdiagm(n, n, 0 => 0.7 * ones(n))
    sum_expected = foo1_native * dx + dx' * foo2_native

    @test norm(sum_native - sum_expected) < 1e-10
    println(io0(), "[TEST] Exact pattern test passed")
end

println(io0(), "[TEST] All tests completed successfully")
