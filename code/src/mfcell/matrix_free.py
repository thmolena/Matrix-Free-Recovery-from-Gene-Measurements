"""Matrix-free graph operators and certified scalar/batched PCG solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg


DENSE_GLOBAL_ALLOCATIONS = 0


@dataclass(frozen=True)
class SolveCertificate:
    iterations: int
    relative_residual: float
    converged: bool


@dataclass(frozen=True)
class BatchCertificate:
    per_column: tuple[SolveCertificate, ...]
    synchronized_iterations: int
    sparse_operator_launches: int

    @property
    def converged(self) -> bool:
        return all(item.converged for item in self.per_column)

    @property
    def max_relative_residual(self) -> float:
        return max(item.relative_residual for item in self.per_column)


def transition_matrix(
    neighbors: np.ndarray,
    distances: np.ndarray,
    distance_scale: float,
) -> sp.csr_matrix:
    """Return a row-stochastic radial transition on supplied neighbor edges."""
    n, k = neighbors.shape
    rows = np.repeat(np.arange(n), k)
    cols = np.asarray(neighbors, dtype=np.int64).reshape(-1)
    distance = np.asarray(distances, dtype=float).reshape(-1)
    scale = max(float(distance_scale), np.finfo(float).tiny)
    weights = np.exp(-(distance * distance) / (2.0 * scale * scale))
    transition = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    row_sum = np.asarray(transition.sum(axis=1)).ravel()
    if np.any(row_sum <= 0.0):
        raise ValueError("every transition row must have positive weight")
    return (sp.diags(1.0 / row_sum) @ transition).tocsr()


def roughness_action(transition: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Apply ``(I-P)^T(I-P)`` to one vector or a dense block of vectors."""
    difference = values - transition @ values
    return difference - transition.T @ difference


def normal_components(
    transition: sp.csr_matrix,
    fit_mask: np.ndarray,
    regularization: float,
    graph_weight: float,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Return mask, Jacobi diagonal, and an action accepting vectors or blocks."""
    mask = np.asarray(fit_mask, dtype=float)
    mu = float(regularization)
    lam = float(graph_weight)
    if mu <= 0.0:
        raise ValueError("regularization must be positive")
    if lam < 0.0:
        raise ValueError("graph_weight must be nonnegative")
    incoming_squared = np.asarray(transition.multiply(transition).sum(axis=0)).ravel()
    diagonal = mask + mu + lam * (1.0 + incoming_squared)

    def apply(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        anchor = mask + mu
        if array.ndim == 2:
            anchor = anchor[:, None]
        return anchor * array + lam * roughness_action(transition, array)

    return mask, diagonal, apply


def normal_action(
    transition: sp.csr_matrix,
    fit_mask: np.ndarray,
    regularization: float,
    graph_weight: float,
) -> tuple[LinearOperator, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Build ``M + mu I + lambda (I-P)^T(I-P)`` as a LinearOperator."""
    _, diagonal, apply = normal_components(
        transition, fit_mask, regularization, graph_weight
    )
    n = transition.shape[0]
    operator = LinearOperator(
        (n, n),
        matvec=apply,
        rmatvec=apply,
        matmat=apply,
        dtype=float,
    )
    return operator, diagonal, lambda x: roughness_action(transition, x)


def solve_residual_field_scalar(
    transition: sp.csr_matrix,
    fit_mask: np.ndarray,
    residual: np.ndarray,
    regularization: float,
    graph_weight: float,
    rtol: float = 1e-9,
    maxiter: int = 1500,
) -> tuple[np.ndarray, list[SolveCertificate]]:
    """Solve each right-hand side independently with SciPy PCG."""
    operator, diagonal, _ = normal_action(
        transition, fit_mask, regularization, graph_weight
    )
    preconditioner = LinearOperator(
        operator.shape,
        matvec=lambda x: x / diagonal,
        rmatvec=lambda x: x / diagonal,
        dtype=float,
    )
    right = np.asarray(fit_mask, dtype=float)[:, None] * np.asarray(residual, dtype=float)
    solution = np.zeros_like(right)
    certificates: list[SolveCertificate] = []
    for column in range(right.shape[1]):
        iterations = 0

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        vector, info = cg(
            operator,
            right[:, column],
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
        )
        solution[:, column] = vector
        relative = float(
            np.linalg.norm(operator @ vector - right[:, column])
            / max(np.linalg.norm(right[:, column]), np.finfo(float).tiny)
        )
        certificates.append(SolveCertificate(iterations, relative, info == 0))
    return solution, certificates


def solve_residual_field_batched(
    transition: sp.csr_matrix,
    fit_mask: np.ndarray,
    residual: np.ndarray,
    regularization: float,
    graph_weight: float,
    rtol: float = 1e-9,
    maxiter: int = 1500,
) -> tuple[np.ndarray, BatchCertificate]:
    """Synchronize independent PCG columns into sparse matrix-block actions.

    Coefficients are computed column by column, so this is not block CG: columns
    never mix.  In exact arithmetic each active column is the corresponding
    scalar Jacobi-PCG trajectory.  Synchronization reduces sparse-operator
    launches while preserving the declared objective and stopping rule.
    """
    mask, diagonal, apply = normal_components(
        transition, fit_mask, regularization, graph_weight
    )
    right = mask[:, None] * np.asarray(residual, dtype=float)
    if right.ndim != 2 or right.shape[0] != transition.shape[0]:
        raise ValueError("residual must have shape (n_vertices, n_targets)")
    solution = np.zeros_like(right)
    residual_now = right.copy()
    preconditioned = residual_now / diagonal[:, None]
    direction = preconditioned.copy()
    rz = np.sum(residual_now * preconditioned, axis=0)
    right_norm = np.linalg.norm(right, axis=0)
    denominator = np.maximum(right_norm, np.finfo(float).tiny)
    active = right_norm > 0.0
    iterations = np.zeros(right.shape[1], dtype=int)
    initial_zero = ~active
    iterations[initial_zero] = 0
    launches = 0
    completed_iteration = 0

    for step in range(1, maxiter + 1):
        if not np.any(active):
            break
        applied = apply(direction)
        launches += 1
        pap = np.sum(direction * applied, axis=0)
        alpha = np.divide(
            rz,
            pap,
            out=np.zeros_like(rz),
            where=active & (np.abs(pap) > np.finfo(float).tiny),
        )
        solution += direction * alpha
        residual_now -= applied * alpha
        relative = np.linalg.norm(residual_now, axis=0) / denominator
        newly_converged = active & (relative <= rtol)
        iterations[newly_converged] = step
        active[newly_converged] = False
        completed_iteration = step
        if not np.any(active):
            break
        preconditioned = residual_now / diagonal[:, None]
        rz_next = np.sum(residual_now * preconditioned, axis=0)
        beta = np.divide(
            rz_next,
            rz,
            out=np.zeros_like(rz),
            where=active & (np.abs(rz) > np.finfo(float).tiny),
        )
        direction = preconditioned + direction * beta
        direction[:, ~active] = 0.0
        rz = rz_next

    iterations[active] = maxiter
    actual_relative = np.linalg.norm(apply(solution) - right, axis=0) / denominator
    launches += 1
    per_column = tuple(
        SolveCertificate(
            int(iterations[column]),
            float(actual_relative[column]),
            bool(initial_zero[column] or actual_relative[column] <= rtol),
        )
        for column in range(right.shape[1])
    )
    return solution, BatchCertificate(
        per_column=per_column,
        synchronized_iterations=completed_iteration,
        sparse_operator_launches=launches,
    )


# Backward-compatible name for the manuscript's primary path.
solve_residual_field = solve_residual_field_batched


def explicit_dense_normal(
    transition: sp.csr_matrix,
    fit_mask: np.ndarray,
    regularization: float,
    graph_weight: float,
) -> np.ndarray:
    """Assemble the dense comparison matrix for bounded equivalence tests only."""
    global DENSE_GLOBAL_ALLOCATIONS
    DENSE_GLOBAL_ALLOCATIONS += 1
    difference = sp.eye(transition.shape[0], format="csr") - transition
    normal = (
        sp.diags(np.asarray(fit_mask, dtype=float) + float(regularization))
        + float(graph_weight) * (difference.T @ difference)
    )
    return normal.toarray()


def sparse_storage_scalars(transition: sp.csr_matrix) -> int:
    """Count numerical values and integer indices/pointers in CSR storage."""
    return int(
        transition.data.size + transition.indices.size + transition.indptr.size
    )
