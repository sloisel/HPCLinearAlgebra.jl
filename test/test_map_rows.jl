using MPI
MPI.Init()

using LinearAlgebraMPI
using LinearAlgebra
using StaticArrays
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

@testset "map_rows" begin
    # Test 1: Single VectorMPI, f returns scalar
    @testset "VectorMPI -> scalar" begin
        v = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = map_rows(r -> r^2, v)
        expected = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
        @test Vector(result) ≈ expected
    end

    # Test 2: Two VectorMPIs, f returns scalar (element-wise product)
    @testset "Two VectorMPIs -> scalar" begin
        u = VectorMPI([1.0, 2.0, 3.0, 4.0])
        v = VectorMPI([4.0, 3.0, 2.0, 1.0])
        result = map_rows((a, b) -> a * b, u, v)
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

    # Test 4: f returns SVector -> MatrixMPI
    @testset "f returns SVector -> MatrixMPI" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
        result = map_rows(r -> SVector(1, 2, 3), A)
        expected = [1 2 3; 1 2 3; 1 2 3]
        @test Matrix(result) == expected
        @test size(result) == (3, 3)
    end

    # Test 5: f returns SVector computed from row
    @testset "f returns SVector from row" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])
        result = map_rows(r -> SVector(sum(r), prod(r)), A)
        expected = [3.0 2.0; 7.0 12.0; 11.0 30.0; 15.0 56.0]
        @test Matrix(result) ≈ expected
        @test size(result) == (4, 2)
    end

    # Test 6: MatrixMPI and VectorMPI together
    @testset "MatrixMPI + VectorMPI -> scalar" begin
        A = MatrixMPI([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])
        w = VectorMPI([1.0, 2.0, 3.0, 4.0])
        # Compute weighted row sums
        result = map_rows((row, wi) -> sum(row) * wi, A, w)
        expected = [6.0 * 1.0, 15.0 * 2.0, 24.0 * 3.0, 33.0 * 4.0]
        @test Vector(result) ≈ expected
    end

    # Test 7: Two MatrixMPIs, f returns scalar
    @testset "Two MatrixMPIs -> scalar" begin
        A = MatrixMPI([1.0 2.0; 3.0 4.0])
        B = MatrixMPI([10.0 20.0; 30.0 40.0])
        result = map_rows((a, b) -> dot(a, b), A, B)
        # Row 1: [1,2] · [10,20] = 10 + 40 = 50
        # Row 2: [3,4] · [30,40] = 90 + 160 = 250
        expected = [50.0, 250.0]
        @test Vector(result) ≈ expected
    end

    # Test 8: Different partitions (repartition should align)
    @testset "Different partitions" begin
        u = VectorMPI([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = VectorMPI([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = map_rows((a, b) -> a + b, u, v)
        expected = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        @test Vector(result) ≈ expected
    end

    # Test 9: Complex numbers
    @testset "Complex numbers" begin
        v = VectorMPI(ComplexF64[1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im])
        result = map_rows(r -> abs2(r), v)
        expected = [5.0, 25.0, 61.0, 113.0]
        @test Vector(result) ≈ expected
    end

    # Test 10: Complex matrix with SVector output
    @testset "Complex matrix -> SVector" begin
        A = MatrixMPI(ComplexF64[1.0+1.0im 2.0-1.0im; 3.0+2.0im 4.0-2.0im])
        result = map_rows(r -> SVector(real(r[1]), imag(r[2])), A)
        expected = [1.0 -1.0; 3.0 -2.0]
        @test Matrix(result) ≈ expected
    end

    # Test 11: Identity transform (SVector pass-through)
    @testset "Identity SVector transform" begin
        A = MatrixMPI([1.0 2.0 3.0; 4.0 5.0 6.0])
        result = map_rows(r -> r, A)  # r is already SVector
        @test Matrix(result) ≈ Matrix(A)
    end

    # Test 12: Scalar from matrix row max
    @testset "Row max" begin
        A = MatrixMPI([1.0 5.0 3.0; 7.0 2.0 4.0; 3.0 3.0 9.0])
        result = map_rows(r -> maximum(r), A)
        expected = [5.0, 7.0, 9.0]
        @test Vector(result) ≈ expected
    end
end

println(LinearAlgebraMPI.io0(), "All map_rows tests passed!")
