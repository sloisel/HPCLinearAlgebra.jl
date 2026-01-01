#!/usr/bin/env julia
#
# Test GPU_MIN_SIZE threshold and mixed-backend operations
#
# Run with:
#   mpiexec -n 1 julia --project=. tools/test_gpu_min_size.jl

using MPI
MPI.Init()

using Metal
using LinearAlgebraMPI
using LinearAlgebraMPI: mtl, cpu, GPU_MIN_SIZE
using LinearAlgebra
using SparseArrays

println("="^60)
println("Testing GPU_MIN_SIZE threshold")
println("="^60)

# Test 1: Small vector stays on CPU
println("\n--- Test 1: Small vector stays on CPU ---")
GPU_MIN_SIZE[] = 1000  # Threshold
small_v = VectorMPI(Float32.(1:100))  # 100 elements < 1000
small_v_mtl = mtl(small_v)
println("  Original type: $(typeof(small_v.v))")
println("  After mtl(): $(typeof(small_v_mtl.v))")
@assert small_v_mtl.v isa Vector "Small vector should stay on CPU"
println("  PASS: Small vector stayed on CPU")

# Test 2: Large vector goes to GPU
println("\n--- Test 2: Large vector goes to GPU ---")
large_v = VectorMPI(Float32.(1:2000))  # 2000 elements > 1000
large_v_mtl = mtl(large_v)
println("  Original type: $(typeof(large_v.v))")
println("  After mtl(): $(typeof(large_v_mtl.v))")
@assert !(large_v_mtl.v isa Vector) "Large vector should go to GPU"
println("  PASS: Large vector went to GPU")

# Test 3: Mixed backend vector operations
println("\n--- Test 3: Mixed backend operations ---")
# Keep threshold high to get mixed backends
GPU_MIN_SIZE[] = 500
v_cpu = VectorMPI(Float32.(1:600))  # Will go to GPU (600 > 500)
v_gpu = mtl(v_cpu)
GPU_MIN_SIZE[] = 1000  # Now new vectors stay on CPU
v_cpu2 = VectorMPI(Float32.(1:600))  # Will stay on CPU (600 < 1000)

println("  v_gpu type: $(typeof(v_gpu.v))")
println("  v_cpu2 type: $(typeof(v_cpu2.v))")

# Test dot product with mixed backends
d = dot(v_gpu, v_cpu2)
println("  dot(gpu, cpu) = $d")
expected = dot(Float32.(1:600), Float32.(1:600))
@assert abs(d - expected) < 1e-3 "Dot product mismatch"
println("  PASS: Mixed dot product works")

# Test addition with mixed backends
sum_vec = v_gpu + v_cpu2
println("  (gpu + cpu) result type: $(typeof(sum_vec.v))")
@assert sum_vec.v isa Vector "Mixed addition should produce CPU result"
println("  PASS: Mixed addition works")

# Test 4: Mixed backend sparse operations
println("\n--- Test 4: Mixed backend sparse operations ---")
A_sparse = sparse([1,2,3], [1,2,3], Float32[1,2,3], 3, 3)
A_cpu = SparseMatrixMPI{Float32}(A_sparse)
GPU_MIN_SIZE[] = 1  # Allow GPU
A_gpu = mtl(A_cpu)
GPU_MIN_SIZE[] = 1000  # New matrices stay on CPU
B_cpu = SparseMatrixMPI{Float32}(A_sparse)

println("  A_gpu nzval type: $(typeof(A_gpu.nzval))")
println("  B_cpu nzval type: $(typeof(B_cpu.nzval))")

# Test sparse * sparse with mixed backends
C = A_gpu * B_cpu
println("  (gpu * cpu) result nzval type: $(typeof(C.nzval))")
@assert C.nzval isa Vector "Mixed sparse multiply should produce CPU result"
println("  PASS: Mixed sparse multiply works")

# Test sparse + sparse with mixed backends
D = A_gpu + B_cpu
println("  (gpu + cpu) result nzval type: $(typeof(D.nzval))")
@assert D.nzval isa Vector "Mixed sparse addition should produce CPU result"
println("  PASS: Mixed sparse addition works")

println("\n" * "="^60)
println("All tests passed!")
println("="^60)
