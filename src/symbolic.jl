"""
Symbolic factorization phase for distributed multifrontal factorization.

This module computes the structural information needed for factorization:
- Fill-reducing ordering (AMD)
- Elimination tree
- Supernodes
- Row structure for each frontal matrix
- MPI rank assignment for supernodes

The symbolic factorization is computed identically on all MPI ranks.
"""

using SparseArrays
using MPI

# ============================================================================
# Fill-Reducing Ordering
# ============================================================================

"""
    amd_ordering(A::SparseMatrixCSC) -> Vector{Int}

Compute an approximate minimum degree ordering for matrix A.
This reduces fill-in during factorization.

For now, uses a simple minimum degree heuristic. A production
implementation would use AMD.jl or call CHOLMOD.
"""
function amd_ordering(A::SparseMatrixCSC)
    n = size(A, 1)

    # Symmetrize pattern
    Asym = A + A'

    # Simple minimum degree algorithm
    # Track degree (number of nonzeros) for each node
    remaining = Set(1:n)
    perm = Vector{Int}(undef, n)

    # Compute initial degrees
    degree = zeros(Int, n)
    for j = 1:n
        for i in nzrange(Asym, j)
            row = rowvals(Asym)[i]
            if row != j
                degree[j] += 1
            end
        end
    end

    # Build adjacency lists
    adj = [Set{Int}() for _ = 1:n]
    for j = 1:n
        for i in nzrange(Asym, j)
            row = rowvals(Asym)[i]
            if row != j
                push!(adj[j], row)
                push!(adj[row], j)
            end
        end
    end

    for k = 1:n
        # Find node with minimum degree among remaining
        min_deg = typemax(Int)
        min_node = 0
        for node in remaining
            if degree[node] < min_deg
                min_deg = degree[node]
                min_node = node
            end
        end

        perm[k] = min_node
        delete!(remaining, min_node)

        # Update degrees: neighbors of min_node become connected
        neighbors = collect(adj[min_node] ∩ remaining)
        for i in 1:length(neighbors)
            for j in i+1:length(neighbors)
                ni, nj = neighbors[i], neighbors[j]
                if !(nj in adj[ni])
                    push!(adj[ni], nj)
                    push!(adj[nj], ni)
                    degree[ni] += 1
                    degree[nj] += 1
                end
            end
        end

        # Remove min_node from neighbor lists
        for neighbor in neighbors
            delete!(adj[neighbor], min_node)
            degree[neighbor] -= 1
        end
    end

    return perm
end

# ============================================================================
# Elimination Tree
# ============================================================================

"""
    elimination_tree(A::SparseMatrixCSC) -> Vector{Int}

Compute the elimination tree of a sparse matrix.

For an unsymmetric matrix, computes the elimination tree of the
symmetric pattern A + A'.

Returns `parent` where `parent[j]` is the parent of node j in the
elimination tree, or 0 if j is a root.

Algorithm from Liu (1990): "The Role of Elimination Trees in Sparse Factorization"
"""
function elimination_tree(A::SparseMatrixCSC)
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square"

    parent = zeros(Int, n)
    ancestor = zeros(Int, n)  # For path compression

    # Work with lower triangular part of A + A'
    # We process column by column
    for k = 1:n
        ancestor[k] = k

        # Process column k: look for entries A[row, k] where row < k
        for i in nzrange(A, k)
            row = rowvals(A)[i]
            if row >= k
                # Skip diagonal and lower triangle entries
                continue
            end
            # row < k: this is in the upper triangle

            # Walk up the tree using path compression
            r = row
            while ancestor[r] != r && ancestor[r] != k
                t = ancestor[r]
                ancestor[r] = k
                r = t
            end
            if ancestor[r] == r
                parent[r] = k
                ancestor[r] = k
            end
        end

        # Also process entries where A[k, j] != 0 for j < k
        # (transposed entries for symmetrization)
        for j = 1:k-1
            for i in nzrange(A, j)
                if rowvals(A)[i] == k
                    # A[k, j] != 0
                    r = j
                    while ancestor[r] != r && ancestor[r] != k
                        t = ancestor[r]
                        ancestor[r] = k
                        r = t
                    end
                    if ancestor[r] == r
                        parent[r] = k
                        ancestor[r] = k
                    end
                    break
                end
            end
        end
    end

    return parent
end

"""
    elimination_tree_sym(A::SparseMatrixCSC) -> Vector{Int}

Compute the elimination tree assuming A is already symmetric and only
the lower triangular part is stored.
"""
function elimination_tree_sym(A::SparseMatrixCSC)
    n = size(A, 1)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)

    for k = 1:n
        ancestor[k] = k
        for i in nzrange(A, k)
            row = rowvals(A)[i]
            if row >= k
                continue  # Only process lower triangle (row < k for column k)
            end

            r = row
            while ancestor[r] != r && ancestor[r] != k
                t = ancestor[r]
                ancestor[r] = k
                r = t
            end
            if ancestor[r] == r
                parent[r] = k
                ancestor[r] = k
            end
        end
    end

    return parent
end

"""
    postorder_etree(parent::Vector{Int}) -> Vector{Int}

Compute a postorder traversal of the elimination tree.

Returns a permutation vector `postorder` such that children always
appear before their parent.
"""
function postorder_etree(parent::Vector{Int})
    n = length(parent)

    # Build children lists
    children = [Int[] for _ = 1:n]
    roots = Int[]

    for i = 1:n
        if parent[i] == 0
            push!(roots, i)
        else
            push!(children[parent[i]], i)
        end
    end

    # DFS postorder traversal
    postorder = Vector{Int}(undef, n)
    idx = 0

    function dfs(node)
        for child in children[node]
            dfs(child)
        end
        idx += 1
        postorder[idx] = node
    end

    for root in roots
        dfs(root)
    end

    return postorder
end

# ============================================================================
# Supernodes
# ============================================================================

"""
    find_supernodes(A, parent, postorder) -> (supernodes, snode_parent)

Detect fundamental supernodes and build the supernodal elimination tree.

A fundamental supernode is a maximal set of contiguous columns [j, j+k]
where:
- Column j+1's pattern = column j's pattern plus row j+1
- parent[j] = j+1 for j in [first, last-1]

Returns:
- `supernodes`: Vector of Supernode objects
- `snode_parent`: Parent supernode for each supernode (0 for roots)
"""
function find_supernodes(A::SparseMatrixCSC, parent::Vector{Int}, postorder::Vector{Int})
    n = size(A, 1)

    # Mark supernode boundaries
    snode_end = falses(n)
    for j = 1:n-1
        if parent[j] != j + 1
            snode_end[j] = true
        end
    end
    snode_end[n] = true

    # Build supernodes
    supernodes = Supernode[]
    node_to_snode = zeros(Int, n)  # Map from column to supernode index

    start = 1
    for j = 1:n
        if snode_end[j]
            push!(supernodes, Supernode(start:j))
            snode_idx = length(supernodes)
            for k = start:j
                node_to_snode[k] = snode_idx
            end
            start = j + 1
        end
    end

    # Build supernodal parent structure
    nsupernodes = length(supernodes)
    snode_parent = zeros(Int, nsupernodes)

    for (sidx, snode) in enumerate(supernodes)
        last_col = last(snode)
        p = parent[last_col]
        if p != 0
            snode_parent[sidx] = node_to_snode[p]
        end
    end

    return supernodes, snode_parent, node_to_snode
end

"""
    supernode_postorder(supernodes, snode_parent) -> Vector{Int}

Compute a postorder traversal of the supernodal tree.
"""
function supernode_postorder(supernodes::Vector{Supernode}, snode_parent::Vector{Int})
    nsupernodes = length(supernodes)

    # Build children lists
    children = [Int[] for _ = 1:nsupernodes]
    roots = Int[]

    for sidx = 1:nsupernodes
        p = snode_parent[sidx]
        if p == 0
            push!(roots, sidx)
        else
            push!(children[p], sidx)
        end
    end

    # DFS postorder
    postorder = Vector{Int}(undef, nsupernodes)
    idx = 0

    function dfs(node)
        for child in children[node]
            dfs(child)
        end
        idx += 1
        postorder[idx] = node
    end

    for root in roots
        dfs(root)
    end

    return postorder
end

# ============================================================================
# Symbolic Factorization
# ============================================================================

"""
    symbolic_factorization(A, supernodes, snode_parent, postorder) -> Vector{FrontalInfo}

Compute the structure of each frontal matrix.

For each supernode, determines:
- Row indices that appear in the frontal matrix
- Number of fully summed rows (equals supernode size)
"""
function symbolic_factorization(A::SparseMatrixCSC{T},
                                supernodes::Vector{Supernode},
                                snode_parent::Vector{Int},
                                postorder::Vector{Int}) where T
    n = size(A, 1)
    nsupernodes = length(supernodes)

    # Map from column to supernode
    col_to_snode = zeros(Int, n)
    for (sidx, snode) in enumerate(supernodes)
        for col in snode.cols
            col_to_snode[col] = sidx
        end
    end

    # Build children lists for supernodes
    snode_children = [Int[] for _ = 1:nsupernodes]
    for sidx = 1:nsupernodes
        p = snode_parent[sidx]
        if p != 0
            push!(snode_children[p], sidx)
        end
    end

    # Compute row structure for each supernode
    # Process in postorder (children before parents)
    frontal_info = Vector{FrontalInfo}(undef, nsupernodes)

    # For each supernode, track which rows appear in its frontal matrix
    snode_rows = [Set{Int}() for _ = 1:nsupernodes]

    for sidx in postorder
        snode = supernodes[sidx]

        # Start with rows from original matrix A
        for col in snode.cols
            # Rows from column of A
            for i in nzrange(A, col)
                row = rowvals(A)[i]
                push!(snode_rows[sidx], row)
            end
            # Rows from row of A (transpose entries)
            for j = 1:n
                for i in nzrange(A, j)
                    if rowvals(A)[i] == col
                        push!(snode_rows[sidx], j)
                    end
                end
            end
        end

        # Add contributions from children (update matrices)
        for child_sidx in snode_children[sidx]
            child_snode = supernodes[child_sidx]
            # Rows in child's update matrix = rows not in child supernode
            for row in snode_rows[child_sidx]
                if !(row in child_snode.cols)
                    push!(snode_rows[sidx], row)
                end
            end
        end

        # Organize row indices: supernode columns first, then update rows
        snode_cols_set = Set(snode.cols)
        all_rows_set = snode_rows[sidx]

        # Fully summed rows = supernode columns (in order)
        fully_summed = collect(snode.cols)

        # Update rows = other rows, sorted
        update = sort([r for r in all_rows_set if !(r in snode_cols_set)])

        # Combined: fully summed first, then update
        all_rows = vcat(fully_summed, update)

        nfs = length(snode.cols)

        frontal_info[sidx] = FrontalInfo(sidx, all_rows, nfs)
    end

    return frontal_info
end

# ============================================================================
# MPI Rank Assignment (MUMPS-style subtree mapping)
# ============================================================================

"""
    assign_supernodes_to_ranks(supernodes, snode_parent, frontal_info, snode_postorder, nranks)
        -> (snode_owner, snode_children)

Assign supernodes to MPI ranks using MUMPS-style subtree mapping with node splitting.

Strategy (following MUMPS approach):
- Small subtrees are assigned entirely to single ranks (minimizes communication)
- Large subtrees near the root are split across ranks for better parallelism
- When a subtree exceeds the split threshold, children are assigned first (potentially
  to different ranks), then the parent is assigned to the least-loaded rank
- This creates cross-rank boundaries that enable parallel factorization and solve

The split threshold is based on total_weight / nranks, ensuring that work is
distributed when subtrees are large enough to benefit from parallelism.

Returns:
- `snode_owner`: Vector of rank assignments (0-indexed) for each supernode
- `snode_children`: Children lists for each supernode
"""
function assign_supernodes_to_ranks(supernodes::Vector{Supernode},
                                    snode_parent::Vector{Int},
                                    frontal_info::Vector{FrontalInfo},
                                    snode_postorder::Vector{Int},
                                    nranks::Int)
    nsupernodes = length(supernodes)
    snode_owner = zeros(Int, nsupernodes)

    # Build children lists
    snode_children = [Int[] for _ in 1:nsupernodes]
    for sidx in 1:nsupernodes
        p = snode_parent[sidx]
        if p != 0
            push!(snode_children[p], sidx)
        end
    end

    # Compute local costs and subtree weights
    local_cost = zeros(Float64, nsupernodes)
    subtree_weight = zeros(Float64, nsupernodes)

    for sidx in snode_postorder  # postorder ensures children processed first
        info = frontal_info[sidx]
        nrows = length(info.row_indices)
        nfs = info.nfs
        # Cost estimate: O(nfs * nrows^2) for partial factorization
        local_cost[sidx] = Float64(nfs) * Float64(nrows)^2
        # Subtree weight includes children
        child_weight = sum(subtree_weight[c] for c in snode_children[sidx]; init=0.0)
        subtree_weight[sidx] = local_cost[sidx] + child_weight
    end

    # Collect subtree for each supernode
    function collect_subtree(root::Int)
        result = Int[root]
        for child in snode_children[root]
            append!(result, collect_subtree(child))
        end
        return result
    end

    # Find tree roots (supernodes with parent = 0)
    roots = [sidx for sidx in 1:nsupernodes if snode_parent[sidx] == 0]

    # Compute total weight and split threshold
    total_weight = sum(subtree_weight[r] for r in roots; init=0.0)

    # Split threshold: subtrees larger than this will be split across ranks
    # Using 1.5x average load allows some imbalance while still splitting large subtrees
    split_threshold = total_weight / nranks * 1.5

    # Track assignments
    assigned = falses(nsupernodes)
    rank_loads = zeros(Float64, nranks)

    # Recursive assignment with splitting for large subtrees
    function assign_subtree!(root::Int)
        if assigned[root]
            return
        end

        sw = subtree_weight[root]
        children = snode_children[root]

        # Decision: split or keep together?
        # Split if:
        # 1. Subtree weight exceeds threshold, AND
        # 2. There are children to split off, AND
        # 3. We have multiple ranks
        should_split = sw > split_threshold && !isempty(children) && nranks > 1

        if should_split
            # Large subtree: recursively assign children first (they may go to different ranks)
            for child in children
                assign_subtree!(child)
            end

            # Then assign this node to the least-loaded rank
            # This may create a cross-rank boundary if children went to different ranks
            target_rank = argmin(rank_loads) - 1  # 0-indexed
            snode_owner[root] = target_rank
            assigned[root] = true
            rank_loads[target_rank + 1] += local_cost[root]
        else
            # Small subtree or leaf: assign entire subtree to one rank
            target_rank = argmin(rank_loads) - 1  # 0-indexed
            subtree = collect_subtree(root)

            for s in subtree
                snode_owner[s] = target_rank
                assigned[s] = true
            end
            rank_loads[target_rank + 1] += sw
        end
    end

    # Process all tree roots
    for root in roots
        assign_subtree!(root)
    end

    return snode_owner, snode_children
end

# ============================================================================
# Complete Symbolic Factorization
# ============================================================================

"""
    compute_symbolic_factorization(A::SparseMatrixMPI{T}; symmetric=false) -> SymbolicFactorization

Compute the complete symbolic factorization for a distributed sparse matrix.

This function:
1. Gathers the matrix structure to all ranks (for symbolic analysis only)
2. Computes AMD ordering
3. Builds elimination tree and supernodes
4. Computes frontal matrix structures
5. Assigns supernodes to MPI ranks

The result is identical on all ranks and can be cached for reuse.
"""
function compute_symbolic_factorization(A::SparseMatrixMPI{T}; symmetric::Bool=false) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    # Gather matrix to all ranks for symbolic analysis
    # (This is acceptable since symbolic analysis is done once per structure)
    A_full = SparseMatrixCSC(A)
    n = size(A_full, 1)

    # Compute fill-reducing ordering
    perm = amd_ordering(A_full)
    invperm = zeros(Int, n)
    for i in 1:n
        invperm[perm[i]] = i
    end

    # Apply permutation: Ap = A[perm, perm]
    Ap = A_full[perm, perm]

    # Compute elimination tree
    parent = symmetric ? elimination_tree_sym(Ap) : elimination_tree(Ap)
    postorder = postorder_etree(parent)

    # Find supernodes
    supernodes, snode_parent, col_to_snode = find_supernodes(Ap, parent, postorder)
    snode_postorder = supernode_postorder(supernodes, snode_parent)

    # Compute frontal matrix structures
    frontal_info = symbolic_factorization(Ap, supernodes, snode_parent, snode_postorder)

    # Assign supernodes to ranks
    snode_owner, snode_children = assign_supernodes_to_ranks(
        supernodes, snode_parent, frontal_info, snode_postorder, nranks)

    # Build elimination order mapping
    # snode.cols are column indices in the REORDERED matrix (Ap = A[perm, perm])
    # global_to_elim maps reordered column index to processing order
    elim_to_global = zeros(Int, n)
    global_to_elim = zeros(Int, n)
    elim_counter = 0
    for sidx in snode_postorder
        snode = supernodes[sidx]
        for col in snode.cols
            elim_counter += 1
            global_to_elim[col] = elim_counter
            elim_to_global[elim_counter] = col
        end
    end

    # Compute structural hash
    structural_hash = _ensure_hash(A)

    return SymbolicFactorization(
        perm,
        invperm,
        supernodes,
        snode_parent,
        snode_children,
        snode_postorder,
        frontal_info,
        snode_owner,
        col_to_snode,
        elim_to_global,
        global_to_elim,
        structural_hash,
        n
    )
end

"""
    get_symbolic_factorization(A::SparseMatrixMPI{T}; symmetric=false) -> SymbolicFactorization

Get the symbolic factorization for a matrix, using cache if available.
"""
function get_symbolic_factorization(A::SparseMatrixMPI{T}; symmetric::Bool=false) where T
    hash = _ensure_hash(A)
    if haskey(_symbolic_cache, hash)
        return _symbolic_cache[hash]
    end
    symbolic = compute_symbolic_factorization(A; symmetric=symmetric)
    _symbolic_cache[hash] = symbolic
    return symbolic
end
