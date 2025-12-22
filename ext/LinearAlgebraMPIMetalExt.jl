"""
    LinearAlgebraMPIMetalExt

Extension module for Metal GPU support in LinearAlgebraMPI.
Provides constructors and operations for MtlArray-backed distributed arrays.
"""
module LinearAlgebraMPIMetalExt

using LinearAlgebraMPI
using Metal
using Adapt

# Re-export for convenience
const MtlVectorMPI{T} = LinearAlgebraMPI.VectorMPI{T,MtlVector{T}}

"""
    mtl(v::LinearAlgebraMPI.VectorMPI)

Convert a CPU VectorMPI to a Metal GPU VectorMPI.
"""
function LinearAlgebraMPI.mtl(v::LinearAlgebraMPI.VectorMPI{T}) where T
    return adapt(MtlArray, v)
end

"""
    cpu(v::LinearAlgebraMPI.VectorMPI{T,<:MtlVector})

Convert a Metal GPU VectorMPI to a CPU VectorMPI.
"""
function LinearAlgebraMPI.cpu(v::LinearAlgebraMPI.VectorMPI{T,<:MtlVector}) where T
    return adapt(Array, v)
end

# Note: Metal.jl already provides adapt_storage methods for MtlArray
# so we don't need to define them here

# Type alias for Metal SparseMatrixMPI
const MtlSparseMatrixMPI{T,Ti} = LinearAlgebraMPI.SparseMatrixMPI{T,Ti,MtlVector{T}}

"""
    mtl(A::LinearAlgebraMPI.SparseMatrixMPI)

Convert a CPU SparseMatrixMPI to a Metal GPU SparseMatrixMPI.
The `nzval` and target structure arrays are moved to GPU.
The CPU structure arrays (`rowptr`, `colval`, partitions) remain on CPU for MPI.
"""
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}) where {T,Ti}
    nzval_gpu = MtlVector(A.nzval)
    # Convert structure arrays to GPU (used by unified SpMV kernel)
    rowptr_target = MtlVector(A.rowptr)
    colval_target = MtlVector(A.colval)
    return LinearAlgebraMPI.SparseMatrixMPI{T,Ti,MtlVector{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A.col_indices,
        A.rowptr,
        A.colval,
        nzval_gpu,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Invalidate cached_transpose (would need to convert too)
        A.cached_symmetric,
        rowptr_target,
        colval_target
    )
end

"""
    cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector})

Convert a Metal GPU SparseMatrixMPI to a CPU SparseMatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.SparseMatrixMPI{T,Ti,<:MtlVector}) where {T,Ti}
    nzval_cpu = Array(A.nzval)
    # For CPU, rowptr_target and colval_target are the same as rowptr and colval
    return LinearAlgebraMPI.SparseMatrixMPI{T,Ti,Vector{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A.col_indices,
        A.rowptr,
        A.colval,
        nzval_cpu,
        A.nrows_local,
        A.ncols_compressed,
        nothing,  # Invalidate cached_transpose
        A.cached_symmetric,
        A.rowptr,  # rowptr_target (same as rowptr for CPU)
        A.colval   # colval_target (same as colval for CPU)
    )
end

# Type alias for Metal MatrixMPI
const MtlMatrixMPI{T} = LinearAlgebraMPI.MatrixMPI{T,MtlMatrix{T}}

"""
    mtl(A::LinearAlgebraMPI.MatrixMPI)

Convert a CPU MatrixMPI to a Metal GPU MatrixMPI.
"""
function LinearAlgebraMPI.mtl(A::LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}) where T
    A_gpu = MtlMatrix(A.A)
    return LinearAlgebraMPI.MatrixMPI{T,MtlMatrix{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_gpu
    )
end

"""
    cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:MtlMatrix})

Convert a Metal GPU MatrixMPI to a CPU MatrixMPI.
"""
function LinearAlgebraMPI.cpu(A::LinearAlgebraMPI.MatrixMPI{T,<:MtlMatrix}) where T
    A_cpu = Array(A.A)
    return LinearAlgebraMPI.MatrixMPI{T,Matrix{T}}(
        A.structural_hash,
        A.row_partition,
        A.col_partition,
        A_cpu
    )
end

# ============================================================================
# MUMPS Factorization Support
# ============================================================================

"""
    _array_to_backend(v::Vector{T}, ::Type{<:MtlVector}) where T

Convert a CPU vector to a Metal GPU vector.
Used by MUMPS factorization for round-trip GPU conversion during solve.
"""
function LinearAlgebraMPI._array_to_backend(v::Vector{T}, ::Type{<:MtlVector}) where T
    return MtlVector(v)
end

"""
    _create_output_like(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:MtlVector}) where T

Create a VectorMPI with MtlVector backend from a CPU VectorMPI.
Used by MUMPS factorization to reconstruct GPU output vectors.
"""
function LinearAlgebraMPI._create_output_like(v::LinearAlgebraMPI.VectorMPI{T,<:Vector}, ::Type{<:MtlVector}) where T
    return LinearAlgebraMPI.mtl(v)
end

# ============================================================================
# MatrixPlan Index Array Support
# ============================================================================

"""
    _index_array_type(::Type{MtlVector{T}}, ::Type{Ti}) where {T,Ti}

Map MtlVector{T} value array type to MtlVector{Ti} index array type.
Used by MatrixPlan to store symbolic index arrays on GPU.
"""
LinearAlgebraMPI._index_array_type(::Type{MtlVector{T}}, ::Type{Ti}) where {T,Ti} = MtlVector{Ti}

"""
    _to_target_backend(v::Vector{Ti}, ::Type{MtlVector{T}}) where {Ti,T}

Convert a CPU index vector to Metal GPU.
Used by SparseMatrixMPI constructors to create GPU structure arrays.
"""
LinearAlgebraMPI._to_target_backend(v::Vector{Ti}, ::Type{<:MtlVector}) where {Ti} = MtlVector(v)

end # module
