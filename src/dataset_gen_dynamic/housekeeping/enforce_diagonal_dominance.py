import numpy as np


def enforce_diagonal_dominance(A: np.ndarray) -> None:
    """Shrinks off-diagonal rows in-place so each row is diagonally dominant."""
    print("Eigenvalues of A:", np.linalg.eigvals(A))
    n = A.shape[0]
    for i in range(n):
        diagonal_entry = A[i,i]
        off_sum = np.sum(np.abs(A[i]))
        off_sum_wo_diagonal_entry = off_sum - diagonal_entry
        if off_sum_wo_diagonal_entry >= diagonal_entry:
            mask = np.ones(n, dtype=bool) #[ True * n]
            mask[i] = False  # remove diagonal entry
            A[i, mask] *= (0.9 * diagonal_entry) / off_sum #reduce non-diagonal entry by a factor and divide by sum of the entire row
    print("Eigenvalues (of A) after enforcing diagonal dominance:", np.linalg.eigvals(A))
