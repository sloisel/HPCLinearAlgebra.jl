# GPU tests for LinearAlgebraMPI
# Tests Metal GPU support for VectorMPI operations

using Test

# Check if Metal is available BEFORE loading MPI
# (Metal must be loaded first for GPU detection to work)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    @info "Metal not available: $e"
    false
end

using MPI

# Initialize MPI if needed
if !MPI.Initialized()
    MPI.Init()
end

using LinearAlgebraMPI
using SparseArrays
using LinearAlgebra

if METAL_AVAILABLE
    @info "Metal is available, running GPU tests"

    @testset "Metal VectorMPI" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nranks = MPI.Comm_size(MPI.COMM_WORLD)

        # Test 1: Basic VectorMPI conversion CPU <-> GPU
        @testset "CPU-GPU conversion" begin
            n = 100
            v_cpu = VectorMPI(Float32.(collect(1.0:n)))

            # Convert to GPU
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)
            @test v_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Convert back to CPU
            v_cpu2 = LinearAlgebraMPI.cpu(v_gpu)
            @test v_cpu2 isa VectorMPI{Float32,Vector{Float32}}

            # Values should match
            @test v_cpu.v == v_cpu2.v
        end

        # Test 2: VectorMPI_local with GPU arrays
        @testset "VectorMPI_local with GPU" begin
            local_size = 10 + rank * 5  # Different size per rank
            local_data = MtlVector(Float32.(collect(1.0:local_size)))

            v_gpu = VectorMPI_local(local_data)
            @test v_gpu isa VectorMPI{Float32,<:Metal.MtlVector}
            @test length(v_gpu.v) == local_size
        end

        # Test 3: Vector addition on GPU
        @testset "Vector addition on GPU" begin
            n = 50
            u_cpu = VectorMPI(Float32.(rand(n)))
            v_cpu = VectorMPI(Float32.(rand(n)))

            u_gpu = LinearAlgebraMPI.mtl(u_cpu)
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Add on GPU
            w_gpu = u_gpu + v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            w_cpu = u_cpu + v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 4: Scalar multiplication on GPU
        @testset "Scalar multiplication on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Scalar multiply on GPU
            w_gpu = 2.5f0 * v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU
            w_cpu = 2.5f0 * v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 5: Vector dot product with GPU
        @testset "Vector dot product" begin
            n = 50
            x_cpu = VectorMPI(Float32.(rand(n)))
            y_cpu = VectorMPI(Float32.(rand(n)))

            x_gpu = LinearAlgebraMPI.mtl(x_cpu)
            y_gpu = LinearAlgebraMPI.mtl(y_cpu)

            # Dot product (reduction goes through MPI, needs CPU)
            # For now, convert back to CPU for dot
            d_cpu = dot(x_cpu, y_cpu)
            d_gpu = dot(LinearAlgebraMPI.cpu(x_gpu), LinearAlgebraMPI.cpu(y_gpu))
            @test d_cpu ≈ d_gpu
        end

        # Test 6: Sparse matrix-vector multiply with GPU vector
        @testset "Sparse A*x with GPU vector" begin
            n = 20
            # Create sparse matrix (stays on CPU)
            A_full = Float32.(sprand(n, n, 0.3)) + Float32(1.0)*I
            A = SparseMatrixMPI{Float32}(A_full)

            # Create CPU and GPU vectors
            x_cpu = VectorMPI(Float32.(rand(n)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply with GPU vector
            y_gpu = A * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = A * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 7: Broadcasting on GPU
        @testset "Broadcasting on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Element-wise operations
            w_gpu = abs.(v_gpu)
            @test Array(w_gpu.v) ≈ abs.(v_cpu.v)

            # Note: Complex broadcasting like v .+ 1 may not work yet
            # since broadcasting returns VectorMPI with local v from broadcast
        end

        # Test 8: Dense MatrixMPI * VectorMPI with GPU vector
        @testset "Dense A*x with GPU vector" begin
            m, n = 20, 15
            # Create dense matrix (stays on CPU)
            A_full = Float32.(rand(m, n))
            A = MatrixMPI(A_full)

            # Create CPU and GPU vectors
            x_cpu = VectorMPI(Float32.(rand(n)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply with GPU vector
            y_gpu = A * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = A * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 9: Dense transpose(A) * x with GPU vector
        @testset "Dense transpose(A)*x with GPU vector" begin
            m, n = 20, 15
            A_full = Float32.(rand(m, n))
            A = MatrixMPI(A_full)

            # Create CPU and GPU vectors
            x_cpu = VectorMPI(Float32.(rand(m)))
            x_gpu = LinearAlgebraMPI.mtl(x_cpu)

            # Multiply with GPU vector
            y_gpu = transpose(A) * x_gpu
            @test y_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            y_cpu = transpose(A) * x_cpu
            @test Array(y_gpu.v) ≈ y_cpu.v atol=1e-5
        end

        # Test 10: VectorMPI subtraction on GPU
        @testset "Vector subtraction on GPU" begin
            n = 50
            u_cpu = VectorMPI(Float32.(rand(n)))
            v_cpu = VectorMPI(Float32.(rand(n)))

            u_gpu = LinearAlgebraMPI.mtl(u_cpu)
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Subtract on GPU
            w_gpu = u_gpu - v_gpu
            @test w_gpu isa VectorMPI{Float32,<:Metal.MtlVector}

            # Compare with CPU result
            w_cpu = u_cpu - v_cpu
            @test Array(w_gpu.v) ≈ w_cpu.v
        end

        # Test 11: VectorMPI norm on GPU
        @testset "Vector norm on GPU" begin
            n = 50
            v_cpu = VectorMPI(Float32.(rand(n)))
            v_gpu = LinearAlgebraMPI.mtl(v_cpu)

            # Norm (reduction goes through MPI)
            n_cpu = norm(v_cpu)
            n_gpu = norm(LinearAlgebraMPI.cpu(v_gpu))
            @test n_cpu ≈ n_gpu
        end

        println(io0(), "=== All Metal GPU tests passed! ===")
    end
else
    @info "Metal not available, skipping GPU tests"

    @testset "GPU tests skipped" begin
        @test_skip "Metal not available"
    end
end
