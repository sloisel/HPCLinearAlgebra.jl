"""
Communication plans for distributed multifrontal factorization.

These plans handle all MPI communication required for:
- Gathering sparse matrix entries to initialize frontal matrices
- Sending update matrices (Schur complements) from children to parents
- Distributing factor entries to appropriate ranks
- Solve phase communication
"""

using MPI
using SparseArrays

# ============================================================================
# Frontal Gather Plan
# ============================================================================

"""
    FrontalGatherPlan{T}

Communication plan for gathering sparse matrix entries to initialize a frontal matrix.

Each rank may contribute entries to the frontal matrix. The owner rank receives
all entries and assembles them into the dense frontal matrix.

MPI tags: 50 for structure, 51 for values
"""
mutable struct FrontalGatherPlan{T}
    snode_idx::Int                       # Supernode this plan is for
    owner_rank::Int                      # Rank that owns this supernode (0-indexed)
    # For sending entries (non-owner ranks)
    dest_rank::Int                       # Where to send (owner_rank, or -1 if we are owner)
    send_entries::Vector{Tuple{Int,Int,T}}  # (row, col, val) entries to send
    send_buf::Vector{T}                  # Pre-allocated send buffer (flattened)
    send_idx_buf::Vector{Int}            # Pre-allocated index buffer (row, col pairs)
    send_req::MPI.Request
    # For receiving entries (owner rank only)
    source_ranks::Vector{Int}            # Ranks we receive from (0-indexed)
    recv_counts::Vector{Int}             # Number of entries from each source
    recv_bufs::Vector{Vector{T}}         # Pre-allocated receive buffers
    recv_idx_bufs::Vector{Vector{Int}}   # Pre-allocated index buffers
    recv_reqs::Vector{MPI.Request}
    # Local entries (on owner rank)
    local_entries::Vector{Tuple{Int,Int,T}}
end

"""
    FrontalGatherPlan{T}(snode_idx, owner_rank) where T

Create an empty frontal gather plan.
"""
function FrontalGatherPlan{T}(snode_idx::Int, owner_rank::Int) where T
    return FrontalGatherPlan{T}(
        snode_idx,
        owner_rank,
        -1,
        Tuple{Int,Int,T}[],
        T[],
        Int[],
        MPI.REQUEST_NULL,
        Int[],
        Int[],
        Vector{T}[],
        Vector{Int}[],
        MPI.Request[],
        Tuple{Int,Int,T}[]
    )
end

# ============================================================================
# Update Plan
# ============================================================================

"""
    UpdatePlan{T}

Communication plan for sending update matrices (Schur complements) from
children to parent supernodes.

When a child supernode finishes factorization, its update matrix needs to
be sent to the parent's owner for assembly via extend-add.

MPI tags: 52 for update matrices
"""
mutable struct UpdatePlan{T}
    child_snode_idx::Int                 # Child supernode
    parent_snode_idx::Int                # Parent supernode
    child_owner::Int                     # Rank owning child (0-indexed)
    parent_owner::Int                    # Rank owning parent (0-indexed)
    update_size::Int                     # Size of update matrix (nrows - nfs)
    update_row_indices::Vector{Int}      # Row indices in update matrix
    send_buf::Vector{T}                  # Pre-allocated send buffer
    recv_buf::Vector{T}                  # Pre-allocated receive buffer
    idx_send_buf::Vector{Int}            # Row indices send buffer
    idx_recv_buf::Vector{Int}            # Row indices receive buffer
    send_req::MPI.Request
    recv_req::MPI.Request
    idx_send_req::MPI.Request
    idx_recv_req::MPI.Request
end

"""
    UpdatePlan{T}(child_sidx, parent_sidx, child_owner, parent_owner, update_size) where T

Create an update plan with pre-allocated buffers.
"""
function UpdatePlan{T}(child_sidx::Int, parent_sidx::Int,
                       child_owner::Int, parent_owner::Int,
                       update_size::Int) where T
    buf_size = update_size * update_size
    return UpdatePlan{T}(
        child_sidx,
        parent_sidx,
        child_owner,
        parent_owner,
        update_size,
        Int[],
        Vector{T}(undef, buf_size),
        Vector{T}(undef, buf_size),
        Vector{Int}(undef, update_size),
        Vector{Int}(undef, update_size),
        MPI.REQUEST_NULL,
        MPI.REQUEST_NULL,
        MPI.REQUEST_NULL,
        MPI.REQUEST_NULL
    )
end

# ============================================================================
# Factorization Plans Collection
# ============================================================================

"""
    FactorizationPlans{T}

Complete set of communication plans for a factorization.

Created once during symbolic phase and reused for numerical factorization.
"""
struct FactorizationPlans{T}
    frontal_gather_plans::Vector{FrontalGatherPlan{T}}
    update_plans::Dict{Tuple{Int,Int}, UpdatePlan{T}}  # (child_sidx, parent_sidx) -> plan
    structural_hash::Blake3Hash
end

# ============================================================================
# Plan Creation
# ============================================================================

"""
    create_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) -> FactorizationPlans{T}

Create all communication plans needed for factorization.
"""
function create_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    nsupernodes = length(symbolic.supernodes)

    # Create frontal gather plans
    frontal_gather_plans = Vector{FrontalGatherPlan{T}}(undef, nsupernodes)
    for sidx in 1:nsupernodes
        frontal_gather_plans[sidx] = create_frontal_gather_plan(A, symbolic, sidx)
    end

    # Create update plans for cross-rank child-parent pairs
    update_plans = Dict{Tuple{Int,Int}, UpdatePlan{T}}()
    for sidx in 1:nsupernodes
        parent_sidx = symbolic.snode_parent[sidx]
        if parent_sidx != 0
            child_owner = symbolic.snode_owner[sidx]
            parent_owner = symbolic.snode_owner[parent_sidx]

            # Always create plan (even for same-rank, for uniformity)
            info = symbolic.frontal_info[sidx]
            update_size = length(info.row_indices) - info.nfs
            if update_size > 0
                plan = UpdatePlan{T}(sidx, parent_sidx, child_owner, parent_owner, update_size)
                update_plans[(sidx, parent_sidx)] = plan
            end
        end
    end

    return FactorizationPlans{T}(
        frontal_gather_plans,
        update_plans,
        symbolic.structural_hash
    )
end

"""
    create_frontal_gather_plan(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization, sidx::Int) -> FrontalGatherPlan{T}

Create a plan for gathering sparse matrix entries to initialize frontal matrix for supernode sidx.
"""
function create_frontal_gather_plan(A::SparseMatrixMPI{T},
                                    symbolic::SymbolicFactorization,
                                    sidx::Int) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    snode = symbolic.supernodes[sidx]
    info = symbolic.frontal_info[sidx]
    owner_rank = symbolic.snode_owner[sidx]

    # Create empty plan
    plan = FrontalGatherPlan{T}(sidx, owner_rank)

    # Determine which rows of A we own and which entries belong to this frontal
    my_row_start = A.row_partition[rank + 1]
    my_row_end = A.row_partition[rank + 2] - 1

    # Build set of row indices in frontal
    frontal_rows_set = Set(info.row_indices)
    snode_cols_set = Set(snode.cols)

    # Apply permutation to get permuted indices
    # frontal works in permuted space (Ap = A[perm, perm])
    perm = symbolic.perm

    # Collect entries from local rows that belong to this frontal
    local_entries = Tuple{Int,Int,T}[]
    AT = A.A.parent

    for local_row in 1:size(AT, 2)
        global_row = my_row_start + local_row - 1
        # Convert to permuted space
        perm_row = symbolic.invperm[global_row]

        if !(perm_row in frontal_rows_set)
            continue
        end

        for idx in AT.colptr[local_row]:(AT.colptr[local_row + 1] - 1)
            local_col = AT.rowval[idx]
            global_col = A.col_indices[local_col]
            # Convert to permuted space
            perm_col = symbolic.invperm[global_col]

            # Include if column is in supernode or row is in supernode (for U part)
            if perm_col in snode_cols_set || perm_row in snode_cols_set
                push!(local_entries, (perm_row, perm_col, AT.nzval[idx]))
            end
        end
    end

    if rank == owner_rank
        # We are the owner - store local entries and prepare to receive from others
        plan.local_entries = local_entries
        plan.dest_rank = -1

        # Exchange entry counts
        my_count = length(local_entries)
        send_counts = zeros(Int, nranks)
        send_counts[rank + 1] = my_count
        recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)

        # Identify who sends to us
        plan.source_ranks = Int[]
        plan.recv_counts = Int[]
        for r in 0:nranks-1
            if r != rank && recv_counts[r + 1] > 0
                push!(plan.source_ranks, r)
                push!(plan.recv_counts, recv_counts[r + 1])
            end
        end

        # Pre-allocate receive buffers
        plan.recv_bufs = [Vector{T}(undef, cnt) for cnt in plan.recv_counts]
        plan.recv_idx_bufs = [Vector{Int}(undef, 2 * cnt) for cnt in plan.recv_counts]
        plan.recv_reqs = Vector{MPI.Request}(undef, length(plan.source_ranks))
    else
        # We are not the owner - prepare to send our entries
        plan.dest_rank = owner_rank
        plan.send_entries = local_entries

        # Pre-allocate send buffers
        n_entries = length(local_entries)
        plan.send_buf = Vector{T}(undef, n_entries)
        plan.send_idx_buf = Vector{Int}(undef, 2 * n_entries)

        # Fill index buffer
        for (i, (row, col, _)) in enumerate(local_entries)
            plan.send_idx_buf[2*i - 1] = row
            plan.send_idx_buf[2*i] = col
        end

        # Notify owner of our count
        my_count = n_entries
        send_counts = zeros(Int, nranks)
        send_counts[owner_rank + 1] = my_count
        MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)
    end

    return plan
end

# ============================================================================
# Plan Execution
# ============================================================================

"""
    execute_frontal_gather!(plan::FrontalGatherPlan{T}, A::SparseMatrixMPI{T}) -> Vector{Tuple{Int,Int,T}}

Execute the frontal gather plan to collect all entries for a frontal matrix.
Returns the collected entries on the owner rank, empty vector on other ranks.
"""
function execute_frontal_gather!(plan::FrontalGatherPlan{T}, A::SparseMatrixMPI{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank == plan.owner_rank
        # Owner: receive entries from other ranks
        # Post receives
        for (i, r) in enumerate(plan.source_ranks)
            plan.recv_reqs[i] = MPI.Irecv!(plan.recv_bufs[i], comm; source=r, tag=51)
        end

        # Wait for all receives
        if !isempty(plan.recv_reqs)
            MPI.Waitall(plan.recv_reqs)
        end

        # Combine local and received entries
        all_entries = copy(plan.local_entries)

        for (i, r) in enumerate(plan.source_ranks)
            # Receive index buffer separately (blocking for simplicity)
            idx_buf = Vector{Int}(undef, length(plan.recv_idx_bufs[i]))
            MPI.Recv!(idx_buf, comm; source=r, tag=50)

            # Parse received entries
            buf = plan.recv_bufs[i]
            for j in 1:length(buf)
                row = idx_buf[2*j - 1]
                col = idx_buf[2*j]
                push!(all_entries, (row, col, buf[j]))
            end
        end

        return all_entries
    else
        # Non-owner: send our entries
        if plan.dest_rank >= 0 && !isempty(plan.send_entries)
            # Fill send buffer with values
            for (i, (_, _, val)) in enumerate(plan.send_entries)
                plan.send_buf[i] = val
            end

            # Send indices first (blocking)
            MPI.Send(plan.send_idx_buf, comm; dest=plan.dest_rank, tag=50)

            # Send values
            plan.send_req = MPI.Isend(plan.send_buf, comm; dest=plan.dest_rank, tag=51)
            MPI.Wait(plan.send_req)
        end

        return Tuple{Int,Int,T}[]
    end
end

"""
    send_update!(plan::UpdatePlan{T}, update::Matrix{T}, update_rows::Vector{Int})

Send an update matrix from child to parent owner.
"""
function send_update!(plan::UpdatePlan{T}, update::Matrix{T}, update_rows::Vector{Int}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank == plan.child_owner && plan.child_owner != plan.parent_owner
        # We own the child and need to send to different rank
        n = size(update, 1)
        @assert n == plan.update_size "Update size mismatch"

        # Copy update matrix to send buffer (column-major)
        copyto!(plan.send_buf, vec(update))

        # Copy row indices
        copyto!(plan.idx_send_buf, update_rows)

        # Send
        plan.idx_send_req = MPI.Isend(plan.idx_send_buf, comm; dest=plan.parent_owner, tag=52)
        plan.send_req = MPI.Isend(plan.send_buf, comm; dest=plan.parent_owner, tag=53)
    end
end

"""
    receive_update!(plan::UpdatePlan{T}) -> (Matrix{T}, Vector{Int})

Receive an update matrix on the parent owner.
Returns the update matrix and its row indices.
"""
function receive_update!(plan::UpdatePlan{T}) where T
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank == plan.parent_owner && plan.child_owner != plan.parent_owner
        # We own the parent and receive from different rank
        n = plan.update_size

        # Receive row indices
        MPI.Recv!(plan.idx_recv_buf, comm; source=plan.child_owner, tag=52)

        # Receive update matrix
        MPI.Recv!(plan.recv_buf, comm; source=plan.child_owner, tag=53)

        # Reshape to matrix
        update = reshape(plan.recv_buf[1:n*n], n, n)
        update_rows = plan.idx_recv_buf[1:n]

        return copy(update), copy(update_rows)
    end

    return Matrix{T}(undef, 0, 0), Int[]
end

"""
    wait_update_sends!(plan::UpdatePlan{T})

Wait for update sends to complete.
"""
function wait_update_sends!(plan::UpdatePlan{T}) where T
    if plan.send_req != MPI.REQUEST_NULL
        MPI.Wait(plan.send_req)
        plan.send_req = MPI.REQUEST_NULL
    end
    if plan.idx_send_req != MPI.REQUEST_NULL
        MPI.Wait(plan.idx_send_req)
        plan.idx_send_req = MPI.REQUEST_NULL
    end
end

# ============================================================================
# Plan Caching
# ============================================================================

"""
Global cache for factorization plans, keyed by (structural hash, element type).
"""
const _factorization_plan_cache = Dict{Tuple{Blake3Hash, DataType}, Any}()

"""
    get_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) -> FactorizationPlans{T}

Get factorization plans, using cache if available.
"""
function get_factorization_plans(A::SparseMatrixMPI{T}, symbolic::SymbolicFactorization) where T
    key = (symbolic.structural_hash, T)
    if haskey(_factorization_plan_cache, key)
        return _factorization_plan_cache[key]::FactorizationPlans{T}
    end
    plans = create_factorization_plans(A, symbolic)
    _factorization_plan_cache[key] = plans
    return plans
end

"""
    clear_factorization_plan_cache!()

Clear the factorization plan cache.
"""
function clear_factorization_plan_cache!()
    empty!(_factorization_plan_cache)
end
