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

end # module
