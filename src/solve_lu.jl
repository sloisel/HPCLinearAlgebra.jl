"""
Distributed solve routines for LU factorization.

Given L * U * x = P * b where:
- L is unit lower triangular (distributed)
- U is upper triangular (distributed)
- P is the row permutation from pivoting

Solve by:
1. Apply fill-reducing permutation: work = b[perm]
2. Apply row pivot permutation: work2[k] = work[row_perm[k]]
3. Forward solve: L * y = work2 (distributed)
4. Backward solve: U * x = y (distributed)
5. Apply inverse permutations
"""

using MPI
using SparseArrays

"""
    solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) -> VectorMPI{T}

Solve the linear system A*x = b using the precomputed LU factorization.
"""
function solve(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    x = VectorMPI(zeros(T, F.symbolic.n); partition=b.partition)
    solve!(x, F, b)
    return x
end

"""
    solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T})

Solve A*x = b in-place using LU factorization.
"""
function solve!(x::VectorMPI{T}, F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = F.symbolic.n

    # Gather b to all ranks for the solve
    # (A more optimized version would do this distributedly)
    b_full = Vector(b)

    # Step 1: Apply fill-reducing permutation
    work = Vector{T}(undef, n)
    for i = 1:n
        work[i] = b_full[F.symbolic.perm[i]]
    end

    # Step 2: Apply row pivot permutation
    work2 = Vector{T}(undef, n)
    for k = 1:n
        work2[k] = work[F.row_perm[k]]
    end

    # Gather L and U to all ranks for the solve
    L_full, U_full = gather_L_U(F)

    # Step 3: Forward solve (in elimination order)
    forward_solve_ordered!(work2, L_full, F.symbolic.elim_to_global)

    # Step 4: Backward solve (in reverse elimination order)
    backward_solve_ordered!(work2, U_full, F.symbolic.elim_to_global)

    # Step 5: Apply inverse row pivot permutation
    for k = 1:n
        work[F.row_perm[k]] = work2[k]
    end

    # Step 6: Apply inverse fill-reducing permutation
    result = Vector{T}(undef, n)
    for i = 1:n
        result[i] = work[F.symbolic.invperm[i]]
    end

    # Distribute result back to VectorMPI
    my_start = x.partition[rank + 1]
    my_end = x.partition[rank + 2] - 1
    for i = my_start:my_end
        x.v[i - my_start + 1] = result[i]
    end

    return x
end

"""
    Base.:\\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T

Solve A*x = b using the backslash operator.
"""
function Base.:\(F::LUFactorizationMPI{T}, b::VectorMPI{T}) where T
    return solve(F, b)
end

# ============================================================================
# Local Solve Routines (used after gathering)
# ============================================================================

"""
    forward_solve_ordered!(x, L, elim_to_global)

Solve L * y = x in-place, overwriting x with y.
Processes columns in ELIMINATION ORDER (given by elim_to_global).
"""
function forward_solve_ordered!(x::AbstractVector{T}, L::SparseMatrixCSC{T},
                                 elim_to_global::Vector{Int}) where T
    n = length(x)

    for k = 1:n
        j = elim_to_global[k]  # Global column index at elimination step k
        xj = x[j]
        if xj != zero(T)
            for i in nzrange(L, j)
                row = rowvals(L)[i]
                if row != j  # Off-diagonal only
                    x[row] -= nonzeros(L)[i] * xj
                end
            end
        end
    end

    return x
end

"""
    backward_solve_ordered!(x, U, elim_to_global)

Solve U * y = x in-place, overwriting x with y.
Processes columns in REVERSE ELIMINATION ORDER.
"""
function backward_solve_ordered!(x::AbstractVector{T}, U::SparseMatrixCSC{T},
                                  elim_to_global::Vector{Int}) where T
    n = length(x)

    for k = n:-1:1
        j = elim_to_global[k]  # Global column index at elimination step k

        # Find diagonal entry
        diag_val = zero(T)
        diag_row = 0
        for i in nzrange(U, j)
            row = rowvals(U)[i]
            val = nonzeros(U)[i]
            if row == j
                diag_val = val
                diag_row = row
                break
            elseif diag_row == 0
                diag_val = val
                diag_row = row
            end
        end

        if diag_row == 0
            for i in nzrange(U, j)
                diag_row = rowvals(U)[i]
                diag_val = nonzeros(U)[i]
                break
            end
        end

        if abs(diag_val) < eps(real(T))
            error("Zero diagonal in U at column $j (step $k)")
        end

        x[diag_row] /= diag_val

        # Update all other rows
        for i in nzrange(U, j)
            row = rowvals(U)[i]
            if row != diag_row
                x[row] -= nonzeros(U)[i] * x[diag_row]
            end
        end
    end

    return x
end

"""
    apply_row_pivots!(x, row_perm)

Apply row permutation from partial pivoting.
"""
function apply_row_pivots!(x::AbstractVector, row_perm::Vector{Int})
    n = length(x)
    temp = similar(x)
    for i = 1:n
        temp[i] = x[row_perm[i]]
    end
    copyto!(x, temp)
    return x
end
