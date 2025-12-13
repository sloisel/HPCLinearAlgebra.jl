"""
Tests for distributed solve functionality.

This tests the MUMPS-style distributed triangular solve that avoids
gathering L/U factors to all ranks.
"""

using MPI
MPI.Init()

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

const TOL = 1e-10

# Create deterministic test matrices
function create_spd_tridiagonal(n::Int)
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [4.0*ones(n); -1.0*ones(n-1); -1.0*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_general_tridiagonal(n::Int)
    I_A = [1:n; 1:n-1; 2:n]
    J_A = [1:n; 2:n; 1:n-1]
    V_A = [2.0*ones(n); -0.5*ones(n-1); -0.8*ones(n-1)]
    return sparse(I_A, J_A, V_A, n, n)
end

function create_2d_laplacian(nx::Int, ny::Int)
    n = nx * ny
    I_A = Int[]
    J_A = Int[]
    V_A = Float64[]

    for i = 1:nx
        for j = 1:ny
            idx = (j-1)*nx + i
            push!(I_A, idx); push!(J_A, idx); push!(V_A, 4.0)
            if i > 1
                push!(I_A, idx); push!(J_A, idx-1); push!(V_A, -1.0)
            end
            if i < nx
                push!(I_A, idx); push!(J_A, idx+1); push!(V_A, -1.0)
            end
            if j > 1
                push!(I_A, idx); push!(J_A, idx-nx); push!(V_A, -1.0)
            end
            if j < ny
                push!(I_A, idx); push!(J_A, idx+nx); push!(V_A, -1.0)
            end
        end
    end

    return sparse(I_A, J_A, V_A, n, n)
end

ts = @testset QuietTestSet "distributed solve" begin

# Test 1: Solve plan initialization
if rank == 0
    println("[test] Solve plan initialization")
    flush(stdout)
end

n = 20
A_full = create_general_tridiagonal(n)
A_mpi = SparseMatrixMPI{Float64}(A_full)

F = lu(A_mpi)

# Get the solve plan (this triggers initialization)
plan = LinearAlgebraMPI.get_or_create_solve_plan(F)

@test plan.initialized == true
@test plan.myrank == rank
@test plan.nranks == nranks

# Check that global_to_local and local_to_global are consistent
for local_idx in 1:length(plan.local_to_global)
    elim_idx = plan.local_to_global[local_idx]
    @test plan.global_to_local[elim_idx] == local_idx
end

# Gather supernode counts from all ranks
local_snode_count = Int32(length(plan.my_supernodes_postorder))
all_snode_counts = MPI.Allgather(local_snode_count, comm)
total_snodes = sum(all_snode_counts)

# Total supernodes should equal length of symbolic.supernodes
@test total_snodes == length(F.symbolic.supernodes)

if rank == 0
    println("  Solve plan initialized successfully")
    println("  Supernode distribution: $all_snode_counts")
    println("  Total supernodes: $total_snodes")
    println("  Subtree roots on rank 0: $(length(plan.subtree_roots))")
end

MPI.Barrier(comm)

# Test 2: Distributed LU solve - small matrix
if rank == 0
    println("[test] Distributed LU solve - small matrix")
    flush(stdout)
end

n_small = 8
A_small_full = create_general_tridiagonal(n_small)
A_small_mpi = SparseMatrixMPI{Float64}(A_small_full)

b_small_full = [1.0 + 0.1*i for i in 1:n_small]
b_small = VectorMPI(b_small_full)

F_small = lu(A_small_mpi)
x_small = solve(F_small, b_small)
x_small_full = Vector(x_small)

residual_small = norm(A_small_full * x_small_full - b_small_full, Inf)

if rank == 0
    println("  LU solve residual: $residual_small")
end

@test residual_small < TOL

MPI.Barrier(comm)

# Test 3: Distributed LU solve - 2D Laplacian
if rank == 0
    println("[test] Distributed LU solve - 2D Laplacian")
    flush(stdout)
end

A_2d_full = create_2d_laplacian(4, 4)  # 16 nodes
n_2d = size(A_2d_full, 1)
A_2d_mpi = SparseMatrixMPI{Float64}(A_2d_full)

b_2d_full = [1.0 + 0.1*i for i in 1:n_2d]
b_2d = VectorMPI(b_2d_full)

F_2d = lu(A_2d_mpi)
x_2d = solve(F_2d, b_2d)
x_2d_full = Vector(x_2d)

residual_2d = norm(A_2d_full * x_2d_full - b_2d_full, Inf)

if rank == 0
    println("  2D Laplacian LU solve residual: $residual_2d")
end

@test residual_2d < TOL

MPI.Barrier(comm)

# Test 4: Distributed LDLT solve
if rank == 0
    println("[test] Distributed LDLT solve")
    flush(stdout)
end

n_ldlt = 12
A_ldlt_full = create_spd_tridiagonal(n_ldlt)
A_ldlt_mpi = SparseMatrixMPI{Float64}(A_ldlt_full)

b_ldlt_full = [1.0 + 0.1*i for i in 1:n_ldlt]
b_ldlt = VectorMPI(b_ldlt_full)

F_ldlt = ldlt(A_ldlt_mpi)
x_ldlt = solve(F_ldlt, b_ldlt)
x_ldlt_full = Vector(x_ldlt)

residual_ldlt = norm(A_ldlt_full * x_ldlt_full - b_ldlt_full, Inf)

if rank == 0
    println("  LDLT solve residual: $residual_ldlt")
end

@test residual_ldlt < TOL

MPI.Barrier(comm)

# Test 5: Distributed LDLT solve - 2D Laplacian
if rank == 0
    println("[test] Distributed LDLT solve - 2D Laplacian")
    flush(stdout)
end

A_ldlt_2d_full = create_2d_laplacian(5, 5)  # 25 nodes
n_ldlt_2d = size(A_ldlt_2d_full, 1)
A_ldlt_2d_mpi = SparseMatrixMPI{Float64}(A_ldlt_2d_full)

b_ldlt_2d_full = [1.0 + 0.1*i for i in 1:n_ldlt_2d]
b_ldlt_2d = VectorMPI(b_ldlt_2d_full)

F_ldlt_2d = ldlt(A_ldlt_2d_mpi)
x_ldlt_2d = solve(F_ldlt_2d, b_ldlt_2d)
x_ldlt_2d_full = Vector(x_ldlt_2d)

residual_ldlt_2d = norm(A_ldlt_2d_full * x_ldlt_2d_full - b_ldlt_2d_full, Inf)

if rank == 0
    println("  2D Laplacian LDLT solve residual: $residual_ldlt_2d")
end

@test residual_ldlt_2d < TOL

MPI.Barrier(comm)

# Test 6: Large matrix to force cross-rank communication
# The cross-rank code paths only execute when subtree roots have parents on other ranks
if rank == 0
    println("[test] Large LU with cross-rank communication")
    flush(stdout)
end

# Use a larger 2D Laplacian that creates multiple subtrees across ranks
A_large_full = create_2d_laplacian(8, 8)  # 64 nodes
n_large = size(A_large_full, 1)
A_large_mpi = SparseMatrixMPI{Float64}(A_large_full)

b_large_full = [1.0 + 0.1*i for i in 1:n_large]
b_large = VectorMPI(b_large_full)

F_large = lu(A_large_mpi)

# Check if we have cross-rank communication (subtree roots with parents on other ranks)
plan = LinearAlgebraMPI.get_or_create_solve_plan(F_large)
local_subtree_roots = Int32(length(plan.subtree_roots))
total_subtree_roots = MPI.Allreduce(local_subtree_roots, +, comm)

if rank == 0
    println("  Total subtree roots across all ranks: $total_subtree_roots")
    println("  (Non-zero means cross-rank communication is tested)")
end

x_large = solve(F_large, b_large)
x_large_full = Vector(x_large)

residual_large = norm(A_large_full * x_large_full - b_large_full, Inf)

if rank == 0
    println("  Large LU solve residual: $residual_large")
end

@test residual_large < TOL

MPI.Barrier(comm)

# Test 7: Even larger LDLT to ensure cross-rank paths are exercised
if rank == 0
    println("[test] Large LDLT with cross-rank communication")
    flush(stdout)
end

A_large_ldlt_full = create_2d_laplacian(10, 10)  # 100 nodes
n_large_ldlt = size(A_large_ldlt_full, 1)
A_large_ldlt_mpi = SparseMatrixMPI{Float64}(A_large_ldlt_full)

b_large_ldlt_full = [1.0 + 0.1*i for i in 1:n_large_ldlt]
b_large_ldlt = VectorMPI(b_large_ldlt_full)

F_large_ldlt = ldlt(A_large_ldlt_mpi)

# Check cross-rank communication for LDLT
plan_ldlt = LinearAlgebraMPI.get_or_create_solve_plan(F_large_ldlt)
local_subtree_roots_ldlt = Int32(length(plan_ldlt.subtree_roots))
total_subtree_roots_ldlt = MPI.Allreduce(local_subtree_roots_ldlt, +, comm)

if rank == 0
    println("  Total subtree roots across all ranks: $total_subtree_roots_ldlt")
end

x_large_ldlt = solve(F_large_ldlt, b_large_ldlt)
x_large_ldlt_full = Vector(x_large_ldlt)

residual_large_ldlt = norm(A_large_ldlt_full * x_large_ldlt_full - b_large_ldlt_full, Inf)

if rank == 0
    println("  Large LDLT solve residual: $residual_large_ldlt")
end

@test residual_large_ldlt < TOL

MPI.Barrier(comm)

end  # QuietTestSet

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    total = sum(global_counts)
    println("\nTest Summary: distributed solve | Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])  Total: $total")
    flush(stdout)
end

MPI.Barrier(comm)
MPI.Finalize()

exit_code = global_counts[2] + global_counts[3] > 0 ? 1 : 0
exit(exit_code)
