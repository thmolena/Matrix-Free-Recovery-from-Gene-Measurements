from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from mfcell import matrix_free
from mfcell.experiment import (
    DATASETS,
    TARGET_IDS,
    _environment_metadata,
    _provenance_scalar,
    load_dataset,
    optimal_nonnegative_blend,
    package_root,
    target_free_log_normalize,
)


def test_environment_provenance_is_stable_json():
    def version_probe():
        return "unused"

    normalized = _provenance_scalar(version_probe)
    assert normalized == f"{version_probe.__module__}.{version_probe.__qualname__}"
    assert "0x" not in normalized
    json.dumps(_environment_metadata(), sort_keys=True)


def test_both_public_archives_have_locked_identity_and_dimensions():
    root = package_root()
    for key, spec in DATASETS.items():
        loaded = load_dataset(root, key)
        assert loaded.counts.shape == spec.shape
        assert loaded.counts.nnz == spec.nonzeros
        assert int(loaded.counts.sum()) == spec.total_counts
        assert len(loaded.gene_ids) == spec.shape[0]
        assert len(loaded.barcodes) == spec.shape[1]


def test_target_free_normalizer_is_invariant_to_target_counts():
    counts = sp.csc_matrix(np.array([[3, 0], [2, 7], [5, 11]], dtype=float))
    before, exposure_before = target_free_log_normalize(counts, np.array([0]))
    changed = counts.copy().tolil()
    changed[0, 0] = 3000
    changed[0, 1] = 9000
    after, exposure_after = target_free_log_normalize(changed.tocsc(), np.array([0]))
    np.testing.assert_allclose(exposure_before, exposure_after)
    np.testing.assert_allclose(before[1:].toarray(), after[1:].toarray())


def _small_problem():
    rng = np.random.default_rng(12)
    n = 48
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack(
        ((np.arange(n) + 1) % n, (np.arange(n) + 2) % n, (np.arange(n) + 7) % n)
    ).reshape(-1)
    values = rng.uniform(0.2, 1.0, size=rows.size)
    transition = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    transition = sp.diags(1.0 / np.asarray(transition.sum(axis=1)).ravel()) @ transition
    mask = np.zeros(n, dtype=bool)
    mask[:30] = True
    residual = rng.normal(size=(n, 4))
    return transition.tocsr(), mask, residual


def test_synchronized_pcg_matches_scalar_and_dense():
    transition, mask, residual = _small_problem()
    before = matrix_free.DENSE_GLOBAL_ALLOCATIONS
    batched, batch_certificate = matrix_free.solve_residual_field_batched(
        transition, mask, residual, 0.03, 0.3, rtol=1e-10
    )
    scalar, scalar_certificates = matrix_free.solve_residual_field_scalar(
        transition, mask, residual, 0.03, 0.3, rtol=1e-10
    )
    assert matrix_free.DENSE_GLOBAL_ALLOCATIONS == before
    dense = matrix_free.explicit_dense_normal(transition, mask, 0.03, 0.3)
    expected = np.linalg.solve(dense, mask[:, None] * residual)
    np.testing.assert_allclose(batched, scalar, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(batched, expected, rtol=1e-8, atol=1e-9)
    assert batch_certificate.converged
    assert all(item.converged for item in scalar_certificates)
    assert batch_certificate.sparse_operator_launches < sum(
        item.iterations + 1 for item in scalar_certificates
    )


def test_analytic_blend_is_box_constrained_least_squares_minimizer():
    rng = np.random.default_rng(4)
    error = rng.normal(size=(50, 3))
    correction = rng.normal(size=(50, 3))
    alpha = optimal_nonnegative_blend(error, correction)
    objective = lambda value: np.sum((error - value * correction) ** 2)
    grid = np.linspace(0.0, 1.0, 10001)
    assert objective(alpha) <= min(objective(value) for value in grid) + 1e-8


def test_locked_evidence_withdraws_old_claim_and_passes_new_gates():
    results = json.loads((package_root() / "results/results.json").read_text())
    contract = results["verification_contract"]
    assert contract["discrete_fields"] == "exact"
    assert contract["floating_relative_tolerance"] == 5e-3
    assert contract["floating_absolute_tolerance"] == 5e-5
    assert contract["method_spearman_status"] == "locked_in_reference_environment"
    assert contract["excluded_fields"] == ["environment", "solver_runtime"]
    assert results["protocol"]["normalization_denominator_excludes_all_target_counts"]
    assert not results["protocol"]["test_targets_used_for_selection"]
    assert results["claim_boundaries"]["worldwide_algorithmic_novelty_claimed"] is False
    assert results["claim_boundaries"]["numerical_tests_are_mathematical_proofs"] is False
    assert set(results["protocol"]["target_ids"]) == set(TARGET_IDS)
    for section in ("primary", "external"):
        block = results[section]
        assert block["metrics"]["method"]["mse"] < block["metrics"]["extra_trees"]["mse"]
        assert block["solver"]["all_converged"]
        assert block["solver"]["max_relative_residual"] < 2e-9
        assert block["operator"]["dense_operator_formed_in_recovery"] is False
        assert all(fold["overlap"] == 0 for fold in block["cross_fitting"])
    assert sum(row["method_minus_extra_trees"] < 0 for row in results["primary"]["gene_metrics"]) == 9
    assert sum(row["method_minus_extra_trees"] < 0 for row in results["external"]["gene_metrics"]) == 11
    assert all(row["relative_solution_difference"] < 1e-7 for row in results["dense_equivalence"])
    assert max(row["relative_solution_difference"] for row in results["solver_runtime"]) < 1e-12
