using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

@testset "map_rows" begin
    # Test 1: Single VectorMPI, f returns scalar
    @testset "VectorMPI -> scalar" begin
        v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = map_rows(r -> r[1]^2, v)
        expected = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
        @test Vector(result) ≈ expected
    end

    # Test 2: Two VectorMPIs, f returns scalar (element-wise product)
    @testset "Two VectorMPIs -> scalar" begin
        u = VectorMPI([1.0, 2.0, 3.0, 4.0])
        v = VectorMPI([4.0, 3.0, 2.0, 1.0])
        result = map_rows((a, b) -> a[1] * b[1], u, v)
        expected = [4.0, 6.0, 6.0, 4.0]
        @test Vector(result) ≈ expected
    end

    # Test 3: Single MatrixMPI, f returns scalar (row norms)
    @testset "MatrixMPI -> scalar (row norms)" begin
        A = MatrixMPI([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0; 1.0 1.0 1.0])
        result = map_rows(r -> norm(r), A)
        expected = [1.0, 2.0, 3.0, sqrt(3.0)]
        @test Vector(result) ≈ expected
    end

    # Test 4: f returns column vector -> vcat into longer VectorMPI
    @testset "f returns column vector (vcat semantics)" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
        result = map_rows(r -> [1, 2, 3], A)
        # Each row produces [1,2,3], vcat gives 9 elements
        expected = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        @test Vector(result) == expected
        @test length(result) == 9
    end

    # Test 5: f returns row vector -> vcat stacks into MatrixMPI
    @testset "f returns row vector (vcat semantics)" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
        result = map_rows(r -> [1 2 3], A)  # 1×3 row matrix
        # Each row produces [1 2 3], vcat stacks into 3×3 matrix
        expected = [1 2 3; 1 2 3; 1 2 3]
        @test Matrix(result) == expected
        @test size(result) == (3, 3)
    end

    # Test 6: f returns adjoint row vector
    @testset "f returns adjoint row vector" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])
        result = map_rows(r -> [1, 2, 3]', A)
        expected = [1 2 3; 1 2 3; 1 2 3; 1 2 3]
        @test Matrix(result) == expected
        @test size(result) == (4, 3)
    end

    # Test 7: MatrixMPI and VectorMPI together
    @testset "MatrixMPI + VectorMPI -> scalar" begin
        A = MatrixMPI([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])
        w = VectorMPI([1.0, 2.0, 3.0, 4.0])
        # Compute weighted row sums
        result = map_rows((row, wi) -> sum(row) * wi[1], A, w)
        expected = [6.0 * 1.0, 15.0 * 2.0, 24.0 * 3.0, 33.0 * 4.0]
        @test Vector(result) ≈ expected
    end

    # Test 8: Two MatrixMPIs, f returns column vector (vcat behavior)
    @testset "Two MatrixMPIs -> column vector (vcat)" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0])
        B = MatrixMPI([10.0 20.0; 30.0 40.0])
        result = map_rows((a, b) -> collect(a) .+ collect(b), A, B)
        # Row 1: [1,2] + [10,20] = [11,22] (column vector)
        # Row 2: [3,4] + [30,40] = [33,44] (column vector)
        # vcat: [11, 22, 33, 44]
        expected = [11.0, 22.0, 33.0, 44.0]
        @test Vector(result) ≈ expected
    end

    # Test 9: Different partitions (repartition should align)
    @testset "Different partitions" begin
        u = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = VectorMPI([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = map_rows((a, b) -> a[1] + b[1], u, v)
        expected = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        @test Vector(result) ≈ expected
    end

    # Test 10: Complex numbers
    @testset "Complex numbers" begin
        v = VectorMPI(ComplexF64[1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im])
        result = map_rows(r -> abs2(r[1]), v)
        expected = [5.0, 25.0, 61.0, 113.0]
        @test Vector(result) ≈ expected
    end

    # Test 11: Variable-length vector output
    @testset "Variable length vectors" begin
        v = VectorMPI([1.0, 2.0, 3.0, 4.0])
        # Each element produces a vector of length equal to its value
        result = map_rows(r -> ones(Int(r[1])), v)
        # 1 one, 2 ones, 3 ones, 4 ones = 10 total
        expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        @test Vector(result) ≈ expected
        @test length(result) == 10
    end

    # Test 12: Lazy transpose wrapper
    @testset "transpose wrapper" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])
        result = map_rows(r -> transpose([1, 2, 3]), A)
        expected = [1 2 3; 1 2 3; 1 2 3; 1 2 3]
        @test Matrix(result) == expected
        @test size(result) == (4, 3)
    end

    # Test 13: conj on real vector (should stay as column vector)
    @testset "conj on real vector" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
        result = map_rows(r -> conj([1, 2, 3]), A)
        # conj of real vector is just the vector itself
        expected = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        @test Vector(result) == expected
        @test length(result) == 9
    end

    # Test 14: conj on complex vector (should stay as column vector)
    @testset "conj on complex vector" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0])
        result = map_rows(r -> conj([1.0+1.0im, 2.0-1.0im]), A)
        # conj flips imaginary part
        expected = [1.0-1.0im, 2.0+1.0im, 1.0-1.0im, 2.0+1.0im]
        @test Vector(result) == expected
    end

    # Test 15: adjoint of complex vector (row vector)
    @testset "adjoint of complex vector" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0])
        result = map_rows(r -> [1.0+1.0im, 2.0-1.0im]', A)
        # adjoint = conj + transpose, so row vector with conjugated values
        expected = [1.0-1.0im 2.0+1.0im; 1.0-1.0im 2.0+1.0im]
        @test Matrix(result) == expected
        @test size(result) == (2, 2)
    end
end

println(LinearAlgebraMPI.io0(), "All map_rows tests passed!")
