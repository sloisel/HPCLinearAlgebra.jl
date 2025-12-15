using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using SparseArrays
using Test
using Random

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Use fixed seed for deterministic random data across all ranks
Random.seed!(42)

# Create test matrices and vectors deterministically
n = 8  # Matrix size
m = 6  # For non-square matrices

# Create a test sparse matrix (n x n)
A_sparse_local = sprand(Float64, n, n, 0.3)
A_sparse_local = A_sparse_local + A_sparse_local' + 2I  # Make it symmetric with positive diagonal
A_sparse = SparseMatrixMPI{Float64}(A_sparse_local)

# Create a test dense matrix (n x m)
B_dense_local = rand(Float64, n, m) .+ 0.1
B_dense = MatrixMPI(B_dense_local)

# Create a test dense matrix (m x n)
C_dense_local = rand(Float64, m, n) .+ 0.1
C_dense = MatrixMPI(C_dense_local)

# Create test vectors
x_local = rand(Float64, n) .+ 0.1
x = VectorMPI(x_local)

y_local = rand(Float64, n) .+ 0.1
y = VectorMPI(y_local)

# Test counter
tests_passed = 0
tests_failed = 0

function test_approx(name, result_mpi, expected, tol=1e-10)
    global tests_passed, tests_failed

    # Convert MPI result to local
    if result_mpi isa VectorMPI
        result_local = Vector(result_mpi)
    elseif result_mpi isa MatrixMPI
        result_local = Matrix(result_mpi)
    elseif result_mpi isa SparseMatrixMPI
        result_local = SparseMatrixCSC(result_mpi)
    else
        result_local = result_mpi
    end

    if expected isa SparseMatrixCSC
        expected = Matrix(expected)
        result_local = result_local isa SparseMatrixCSC ? Matrix(result_local) : result_local
    end

    diff = maximum(abs.(result_local .- expected))
    if diff < tol
        tests_passed += 1
        println(io0(), "✓ $name (diff=$diff)")
    else
        tests_failed += 1
        println(io0(), "✗ $name (diff=$diff)")
    end
end

println(io0(), "\n=== Testing New LinearAlgebraMPI Operations ===\n")

# ============================================================================
# Test 1: transpose(SparseMatrixMPI) * VectorMPI
# ============================================================================
println(io0(), "--- transpose(SparseMatrixMPI) * VectorMPI ---")
result1 = transpose(A_sparse) * x
expected1 = transpose(A_sparse_local) * x_local
test_approx("transpose(Sparse) * Vector", result1, expected1)

# ============================================================================
# Test 2: SparseMatrixMPI * MatrixMPI
# ============================================================================
println(io0(), "\n--- SparseMatrixMPI * MatrixMPI ---")
result2 = A_sparse * B_dense
expected2 = A_sparse_local * B_dense_local
test_approx("Sparse * Dense Matrix", result2, expected2)

# ============================================================================
# Test 3: transpose(SparseMatrixMPI) * MatrixMPI
# ============================================================================
println(io0(), "\n--- transpose(SparseMatrixMPI) * MatrixMPI ---")
result3 = transpose(A_sparse) * B_dense
expected3 = transpose(A_sparse_local) * B_dense_local
test_approx("transpose(Sparse) * Dense", result3, expected3)

# ============================================================================
# Test 4: MatrixMPI * SparseMatrixMPI
# ============================================================================
println(io0(), "\n--- MatrixMPI * SparseMatrixMPI ---")
# Need dense matrix with rows that match sparse matrix columns
# E_dense has n rows, D_sparse has n columns, result is n x p
Random.seed!(123)
D_sparse_local = sprand(Float64, n, m, 0.3)
D_sparse_local = D_sparse_local + sparse(1:min(n,m), 1:min(n,m), fill(0.5, min(n,m)), n, m)
D_sparse = SparseMatrixMPI{Float64}(D_sparse_local)

# E_dense is p x n (where p=m), D_sparse is n x m, so E_dense * D_sparse is p x m
Random.seed!(124)
E_dense_local = rand(Float64, m, n) .+ 0.1
E_dense = MatrixMPI(E_dense_local)

result4 = E_dense * D_sparse
expected4 = E_dense_local * D_sparse_local
test_approx("Dense * Sparse", result4, expected4)

# ============================================================================
# Test 5: transpose(MatrixMPI) * MatrixMPI
# ============================================================================
println(io0(), "\n--- transpose(MatrixMPI) * MatrixMPI ---")
# transpose(B_dense) is m x n, B_dense is n x m, so result is m x m
result5 = transpose(B_dense) * B_dense
expected5 = transpose(B_dense_local) * B_dense_local
test_approx("transpose(Dense) * Dense", result5, expected5)

# ============================================================================
# Test 6: transpose(MatrixMPI) * SparseMatrixMPI
# ============================================================================
println(io0(), "\n--- transpose(MatrixMPI) * SparseMatrixMPI ---")
# transpose(B_dense) is m x n, need sparse n x p
# Use A_sparse which is n x n
result6 = transpose(B_dense) * A_sparse
expected6 = transpose(B_dense_local) * A_sparse_local
test_approx("transpose(Dense) * Sparse", result6, expected6)

# ============================================================================
# Test 7: MatrixMPI[:, k] column indexing
# ============================================================================
println(io0(), "\n--- MatrixMPI[:, k] column indexing ---")
for k in 1:m
    result7 = B_dense[:, k]
    expected7 = B_dense_local[:, k]
    test_approx("Dense[:, $k]", result7, expected7)
end

# ============================================================================
# Test 8: SparseMatrixMPI[:, k] column indexing
# ============================================================================
println(io0(), "\n--- SparseMatrixMPI[:, k] column indexing ---")
for k in 1:n
    result8 = A_sparse[:, k]
    expected8 = Vector(A_sparse_local[:, k])
    test_approx("Sparse[:, $k]", result8, expected8)
end

# ============================================================================
# Test 9: dot(VectorMPI, VectorMPI)
# ============================================================================
println(io0(), "\n--- dot(VectorMPI, VectorMPI) ---")
result9 = dot(x, y)
expected9 = dot(x_local, y_local)
if abs(result9 - expected9) < 1e-10
    tests_passed += 1
    println(io0(), "✓ dot(x, y) (diff=$(abs(result9 - expected9)))")
else
    tests_failed += 1
    println(io0(), "✗ dot(x, y) (diff=$(abs(result9 - expected9)))")
end

# Self dot product
result9b = dot(x, x)
expected9b = dot(x_local, x_local)
if abs(result9b - expected9b) < 1e-10
    tests_passed += 1
    println(io0(), "✓ dot(x, x) (diff=$(abs(result9b - expected9b)))")
else
    tests_failed += 1
    println(io0(), "✗ dot(x, x) (diff=$(abs(result9b - expected9b)))")
end

# ============================================================================
# Test 10: UniformScaling A + λI
# ============================================================================
println(io0(), "\n--- UniformScaling A + λI ---")
λ = 3.5
result10 = A_sparse + λ*I
expected10 = A_sparse_local + λ*I
test_approx("A + λI", result10, expected10)

# ============================================================================
# Test 11: UniformScaling A - λI
# ============================================================================
println(io0(), "\n--- UniformScaling A - λI ---")
result11 = A_sparse - λ*I
expected11 = A_sparse_local - λ*I
test_approx("A - λI", result11, expected11)

# ============================================================================
# Test 12: UniformScaling λI + A
# ============================================================================
println(io0(), "\n--- UniformScaling λI + A ---")
result12 = λ*I + A_sparse
expected12 = λ*I + A_sparse_local
test_approx("λI + A", result12, expected12)

# ============================================================================
# Test 13: UniformScaling λI - A
# ============================================================================
println(io0(), "\n--- UniformScaling λI - A ---")
result13 = λ*I - A_sparse
expected13 = λ*I - A_sparse_local
test_approx("λI - A", result13, expected13)

# ============================================================================
# Summary
# ============================================================================
println(io0(), "\n=== Summary ===")
println(io0(), "Tests passed: $tests_passed")
println(io0(), "Tests failed: $tests_failed")

@test tests_failed == 0
