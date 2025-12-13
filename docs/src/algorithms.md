# Algorithms

This document describes the algorithms used in LinearAlgebraMPI's distributed sparse direct solvers. It is intended for experts familiar with sparse matrix computations who want to understand our implementation choices.

## Overview

LinearAlgebraMPI implements distributed sparse LU and LDLT factorizations using the **multifrontal method**. The implementation follows a three-phase approach:

1. **Symbolic Phase**: Compute fill-reducing ordering, elimination tree, supernodes, and rank assignments
2. **Numerical Phase**: Distributed multifrontal factorization with communication
3. **Solve Phase**: Distributed triangular solves

## The Multifrontal Method

The multifrontal method was introduced by Duff and Reid [1] for symmetric indefinite systems and later extended to unsymmetric systems [2]. Liu provides an excellent survey of the method [3].

The key insight is that sparse Gaussian elimination can be organized as a sequence of partial factorizations of dense submatrices called **frontal matrices**. Each frontal matrix corresponds to eliminating a set of pivot columns and produces:
- Completed rows/columns of the factors L and U (or L and D for LDLT)
- A dense **update matrix** (Schur complement) that is passed to a parent frontal

This organization enables:
- High performance through dense BLAS operations on frontal matrices
- Natural parallelism from the tree structure of dependencies
- Reduced memory traffic compared to left-looking or right-looking methods

### Frontal Matrix Structure

For a supernode with `nfs` fully-summed (pivot) columns and `nrs` rows in total, the frontal matrix F has the block structure:

```
F = [ F11  F12 ]    (nfs × nfs)  (nfs × (nrs-nfs))
    [ F21  F22 ]    ((nrs-nfs) × nfs)  ((nrs-nfs) × (nrs-nfs))
```

After partial factorization:
- `F11` contains the diagonal block of the factors
- `F21` contains the subdiagonal of L (LU) or is symmetric with F12 (LDLT)
- `F12` contains the superdiagonal of U (LU only)
- `F22` becomes the update matrix (Schur complement) passed to the parent

### Assembly (Extend-Add)

When a frontal matrix receives update matrices from its children in the elimination tree, these are assembled using the **extend-add** operation. The child's update matrix is scattered into the parent's frontal matrix at positions determined by the index mapping, and values are summed.

## Fill-Reducing Ordering

We use the **Approximate Minimum Degree (AMD)** ordering algorithm [4, 5] to reduce fill-in during factorization. AMD computes a permutation P such that factoring P'AP produces fewer nonzeros in L and U than factoring A directly.

The minimum degree algorithm is a greedy heuristic that, at each elimination step, selects the variable whose elimination creates the least fill-in. AMD uses quotient graph techniques to efficiently approximate the true minimum degree, achieving O(n²) worst-case complexity while producing orderings comparable to exact minimum degree.

Our implementation uses Julia's `AMD.jl` package, which provides a native Julia implementation of the AMD algorithm.

## Elimination Tree

The **elimination tree** captures the dependencies between columns during factorization [6]. For a matrix A with Cholesky factor L, the elimination tree T has:
- n nodes (one per column)
- An edge from j to parent(j) where parent(j) is the row index of the first subdiagonal nonzero in column j of L

Key properties:
- Column j of L depends only on columns in the subtree rooted at j
- Disjoint subtrees can be factored independently (parallelism)
- The tree structure determines the flow of update matrices in the multifrontal method

For unsymmetric LU factorization, we use the elimination tree of the symmetrized structure A + A'.

## Supernodes

A **supernode** is a set of contiguous columns with nearly identical sparsity structure [7, 8]. Grouping columns into supernodes enables:
- Dense matrix operations (BLAS-3) instead of sparse column operations
- Reduced indexing overhead
- Better cache utilization

We detect **fundamental supernodes**: maximal sets of contiguous columns where each column's structure is contained in the next column's structure (plus the new diagonal). This can be characterized in terms of the elimination tree: columns j and j+1 are in the same fundamental supernode if and only if parent(j) = j+1 in the elimination tree [7].

After detecting fundamental supernodes, we construct a **supernodal elimination tree** where each node represents a supernode rather than a single column.

## Distributed Execution

### Supernode-to-Rank Assignment

We use a **subtree-to-rank mapping** strategy inspired by MUMPS [9, 10]. The key idea is to assign complete subtrees of the supernodal elimination tree to single MPI ranks, minimizing communication.

The assignment algorithm:
1. Compute the "work" for each supernode: `work[s] = nfs * nrows² + Σ work[children]`
2. Identify subtree roots (supernodes whose parent is assigned to a different rank or is the root)
3. Use bin-packing to assign subtrees to ranks, balancing the total work per rank

This approach ensures that:
- Communication only occurs at subtree boundaries
- Load is approximately balanced across ranks
- Small problems use fewer ranks efficiently

For very large frontal matrices near the root of the tree, more sophisticated 2D distribution schemes (as in MUMPS [9]) could be employed, but our current implementation assigns each supernode to a single rank.

### Communication Pattern

During numerical factorization, communication occurs when:
1. A child supernode on rank r₁ sends its update matrix to a parent supernode on rank r₂ ≠ r₁
2. The update matrix and its row indices are sent via MPI point-to-point communication

We use non-blocking sends (`MPI.Isend`) to overlap communication with computation where possible. Synchronization barriers between supernodes ensure correct ordering.

## Numerical Pivoting

### LU Factorization: Partial Pivoting

For unsymmetric matrices, we use **partial pivoting** within each frontal matrix. At each elimination step k within the frontal:
1. Find the largest magnitude element in column k below the diagonal
2. Swap rows if necessary
3. Proceed with elimination

The row permutation is tracked and applied during the solve phase. Partial pivoting ensures numerical stability with element growth bounded by 2^(n-1) in the worst case, though growth is typically much smaller in practice.

### LDLT Factorization: Bunch-Kaufman Pivoting

For symmetric matrices (definite or indefinite), we use the **Bunch-Kaufman pivoting** strategy [11]. This produces a factorization:

```
P' * A * P = L * D * L'
```

where:
- P is a permutation matrix
- L is unit lower triangular
- D is block diagonal with 1×1 and 2×2 blocks

The algorithm maintains symmetry while providing numerical stability for indefinite matrices. At each step, it chooses between:
- A 1×1 pivot if the diagonal element is sufficiently large
- A 2×2 pivot using an off-diagonal element if better conditioning is achieved

The threshold parameter α = (1 + √17)/8 ≈ 0.6404 balances stability and sparsity [11].

For 2×2 pivots at positions (k, k+1), the block:
```
D_block = [ d_kk    d_k,k+1  ]
          [ d_k+1,k  d_k+1,k+1 ]
```
is stored, and the solve phase handles these blocks specially by solving 2×2 systems.

**Important**: Our LDLT uses transpose (L'), not conjugate transpose (L*). This is correct for real symmetric and complex symmetric matrices, but NOT for complex Hermitian matrices.

## Solve Phase

### LU Solve

Given the factorization P_r' * L * U * P_c = A (with row permutation P_r from pivoting and column permutation P_c from AMD ordering), we solve Ax = b as:

1. Apply AMD permutation: `y = P_c' * b`
2. Apply row pivot permutation: `z = P_r * y`
3. Forward solve: `L * w = z` (in elimination order)
4. Backward solve: `U * v = w` (in reverse elimination order)
5. Apply inverse row permutation: `u = P_r' * v`
6. Apply inverse AMD permutation: `x = P_c * u`

### LDLT Solve

Given P' * A * P = L * D * L' with symmetric pivot permutation P_s, we solve Ax = b as:

1. Apply AMD permutation: `y = P_perm * b`
2. Apply symmetric pivot permutation: `z = P_s * y`
3. Forward solve: `L * w = z`
4. Diagonal solve: `D * v = w` (handling 2×2 blocks)
5. Backward solve: `L' * u = v`
6. Apply inverse symmetric permutation: `t = P_s' * u`
7. Apply inverse AMD permutation: `x = P_perm' * t`

The diagonal solve for 2×2 blocks requires solving:
```
[ d11  d12 ] [ v_k   ]   [ w_k   ]
[ d21  d22 ] [ v_k+1 ] = [ w_k+1 ]
```
which is done by direct 2×2 matrix inversion.

## Plan Caching

Following the pattern used throughout LinearAlgebraMPI, we cache computed structures for reuse:

- **Symbolic factorization cache**: Keyed by the structural hash of the matrix. Stores the AMD ordering, elimination tree, supernodes, and rank assignments.
- **Factorization plan cache**: Keyed by (structural hash, element type). Stores pre-allocated communication buffers.

When factoring a sequence of matrices with the same sparsity pattern, only the first factorization incurs the cost of symbolic analysis. Subsequent factorizations reuse the cached symbolic structure and communication plans, performing only the numerical computation.

## Complexity

For a sparse matrix of dimension n with nnz nonzeros:

- **AMD ordering**: O(nnz) average case, O(n²) worst case
- **Symbolic factorization**: O(nnz_L) where nnz_L is nonzeros in L
- **Numerical factorization**: Dominated by dense operations on frontal matrices. For 2D problems from finite elements, typically O(n^1.5); for 3D problems, O(n²)
- **Solve**: O(nnz_L + nnz_U)

The distributed implementation adds communication overhead proportional to the number of cross-rank edges in the supernodal elimination tree.

## References

[1] I. S. Duff and J. K. Reid, "The multifrontal solution of indefinite sparse symmetric linear equations," *ACM Transactions on Mathematical Software*, vol. 9, no. 3, pp. 302–325, 1983. [https://doi.org/10.1145/356044.356047](https://doi.org/10.1145/356044.356047)

[2] I. S. Duff and J. K. Reid, "The multifrontal solution of unsymmetric sets of linear equations," *SIAM Journal on Scientific and Statistical Computing*, vol. 5, no. 3, pp. 633–641, 1984.

[3] J. W. H. Liu, "The multifrontal method for sparse matrix solution: Theory and practice," *SIAM Review*, vol. 34, no. 1, pp. 82–109, 1992. [https://doi.org/10.1137/1034004](https://doi.org/10.1137/1034004)

[4] P. R. Amestoy, T. A. Davis, and I. S. Duff, "An approximate minimum degree ordering algorithm," *SIAM Journal on Matrix Analysis and Applications*, vol. 17, no. 4, pp. 886–905, 1996. [https://doi.org/10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952)

[5] P. R. Amestoy, T. A. Davis, and I. S. Duff, "Algorithm 837: AMD, an approximate minimum degree ordering algorithm," *ACM Transactions on Mathematical Software*, vol. 30, no. 3, pp. 381–388, 2004. [https://doi.org/10.1145/1024074.1024081](https://doi.org/10.1145/1024074.1024081)

[6] J. W. H. Liu, "The role of elimination trees in sparse factorization," *SIAM Journal on Matrix Analysis and Applications*, vol. 11, no. 1, pp. 134–172, 1990. [https://doi.org/10.1137/0611010](https://doi.org/10.1137/0611010)

[7] J. W. H. Liu, E. G. Ng, and B. W. Peyton, "On finding supernodes for sparse matrix computations," *SIAM Journal on Matrix Analysis and Applications*, vol. 14, no. 1, pp. 242–252, 1993. [https://doi.org/10.1137/0614019](https://doi.org/10.1137/0614019)

[8] C. Ashcraft and R. Grimes, "The influence of relaxed supernode partitions on the multifrontal method," *ACM Transactions on Mathematical Software*, vol. 15, no. 4, pp. 291–309, 1989. [https://doi.org/10.1145/76909.76912](https://doi.org/10.1145/76909.76912)

[9] P. R. Amestoy, I. S. Duff, J. Koster, and J.-Y. L'Excellent, "A fully asynchronous multifrontal solver using distributed dynamic scheduling," *SIAM Journal on Matrix Analysis and Applications*, vol. 23, no. 1, pp. 15–41, 2001. [https://doi.org/10.1137/S0895479899358194](https://doi.org/10.1137/S0895479899358194)

[10] P. R. Amestoy, I. S. Duff, and J.-Y. L'Excellent, "Multifrontal parallel distributed symmetric and unsymmetric solvers," *Computer Methods in Applied Mechanics and Engineering*, vol. 184, no. 2–4, pp. 501–520, 2000.

[11] J. R. Bunch and L. Kaufman, "Some stable methods for calculating inertia and solving symmetric linear systems," *Mathematics of Computation*, vol. 31, no. 137, pp. 163–179, 1977. [https://doi.org/10.1090/S0025-5718-1977-0428694-0](https://doi.org/10.1090/S0025-5718-1977-0428694-0)
