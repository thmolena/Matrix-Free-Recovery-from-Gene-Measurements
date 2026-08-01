"""Leakage-audited PBMC recovery experiments and manuscript artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import statistics
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = "1"

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_info

from .matrix_free import (
    explicit_dense_normal,
    solve_residual_field_batched,
    solve_residual_field_scalar,
    sparse_storage_scalars,
    transition_matrix,
)


SEED = 20260729
N_INPUT_GENES = 96
N_COMPONENTS = 16
N_NEIGHBORS = 15
REGULARIZATION = 0.03
GRAPH_WEIGHTS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_TREES = 256
N_FOLDS = 5
FINAL_RTOL = 1e-9
# The registered PyPI/macOS reference environment can still vary at the final
# floating-point digits across clean installs. Exact strings, indices,
# selections, hashes, and booleans are compared bit for bit; floating evidence
# uses the declared reference-environment envelope below.
LOCKED_RESULT_REL_TOL = 5e-3
LOCKED_RESULT_ABS_TOL = 5e-5
REFERENCE_PYTHON = (3, 13, 12)
REFERENCE_SYSTEM = "Darwin"
REFERENCE_MACHINE = "arm64"
REFERENCE_BLAS = "accelerate"
REFERENCE_MACOS = "26.6"
REFERENCE_LOCK_SHA256 = "b1b02d84e0991707d67530775a1c01750ce982c8c6a12f30757e23e8a6ad9db9"
REFERENCE_DISTRIBUTIONS = {
    "contourpy": "1.3.3",
    "cycler": "0.12.1",
    "fonttools": "4.63.0",
    "joblib": "1.5.3",
    "kiwisolver": "1.5.0",
    "matplotlib": "3.10.8",
    "numpy": "2.4.3",
    "packaging": "26.2",
    "pillow": "12.3.0",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
}

TARGETS = (
    ("ENSG00000167286", "CD3D"),
    ("ENSG00000168685", "IL7R"),
    ("ENSG00000204482", "LST1"),
    ("ENSG00000156738", "MS4A1"),
    ("ENSG00000115523", "GNLY"),
    ("ENSG00000105374", "NKG7"),
    ("ENSG00000203747", "FCGR3A"),
    ("ENSG00000104972", "LILRB1"),
    ("ENSG00000163736", "PPBP"),
    ("ENSG00000179639", "FCER1A"),
    ("ENSG00000101439", "CST3"),
    ("ENSG00000143546", "S100A8"),
)
TARGET_IDS = tuple(item[0] for item in TARGETS)
TARGET_NAMES = tuple(item[1] for item in TARGETS)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    archive: str
    member_root: str
    url: str
    landing_page: str
    sha256: str
    shape: tuple[int, int]
    nonzeros: int
    total_counts: int
    license: str = "CC BY 4.0"


DATASETS = {
    "pbmc3k": DatasetSpec(
        key="pbmc3k",
        title="10x Genomics PBMC3k",
        archive="pbmc3k_filtered_gene_bc_matrices.tar.gz",
        member_root="filtered_gene_bc_matrices/hg19/",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
            "pbmc3k_filtered_gene_bc_matrices.tar.gz"
        ),
        landing_page=(
            "https://www.10xgenomics.com/datasets/"
            "3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0"
        ),
        sha256="847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5",
        shape=(32738, 2700),
        nonzeros=2286884,
        total_counts=6390631,
    ),
    "pbmc4k": DatasetSpec(
        key="pbmc4k",
        title="10x Genomics PBMC4k",
        archive="pbmc4k_filtered_gene_bc_matrices.tar.gz",
        member_root="filtered_gene_bc_matrices/GRCh38/",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/"
            "pbmc4k_filtered_gene_bc_matrices.tar.gz"
        ),
        landing_page=(
            "https://www.10xgenomics.com/datasets/"
            "4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0"
        ),
        sha256="097ff873bae645bec5586898ac002ccebbce952d52ac19ea0a35c27c4820df26",
        shape=(33694, 4340),
        nonzeros=5727695,
        total_counts=18874824,
    ),
}


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    counts: sp.csc_matrix
    gene_ids: list[str]
    gene_names: list[str]
    barcodes: list[str]


@dataclass
class PreparedDataset:
    x: np.ndarray
    y: np.ndarray
    embedding: np.ndarray
    train: np.ndarray
    test: np.ndarray
    train_mask: np.ndarray
    input_ids: list[str]
    input_names: list[str]
    target_names: list[str]
    target_free_library: np.ndarray


def package_root() -> Path:
    installed = Path(__file__).resolve().parent
    source_tree = Path(__file__).resolve().parents[2]
    return source_tree if (source_tree / "data").is_dir() else installed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"missing archive member {name}")
    return member.read()


def load_dataset(root: Path, key: str) -> LoadedDataset:
    spec = DATASETS[key]
    path = Path(root) / "data" / "source" / spec.archive
    observed = sha256(path)
    if observed != spec.sha256:
        raise ValueError(f"{key} SHA-256 mismatch: {observed}")
    with tarfile.open(path, "r:gz") as archive:
        matrix_bytes = _read_member(archive, spec.member_root + "matrix.mtx")
        genes_bytes = _read_member(archive, spec.member_root + "genes.tsv")
        barcode_bytes = _read_member(archive, spec.member_root + "barcodes.tsv")
    try:
        counts = scipy.io.mmread(io.BytesIO(matrix_bytes), spmatrix=True).tocsc()
    except TypeError:
        counts = scipy.io.mmread(io.BytesIO(matrix_bytes)).tocsc()
    rows = [line.split("\t") for line in genes_bytes.decode().strip().splitlines()]
    gene_ids = [row[0] for row in rows]
    gene_names = [row[1] for row in rows]
    barcodes = barcode_bytes.decode().strip().splitlines()
    if counts.shape != spec.shape:
        raise ValueError(f"{key} shape {counts.shape} != {spec.shape}")
    if counts.nnz != spec.nonzeros:
        raise ValueError(f"{key} nonzeros {counts.nnz} != {spec.nonzeros}")
    if int(counts.sum()) != spec.total_counts:
        raise ValueError(f"{key} molecule count mismatch")
    if len(gene_ids) != counts.shape[0] or len(barcodes) != counts.shape[1]:
        raise ValueError(f"{key} metadata dimensions do not match matrix")
    return LoadedDataset(spec, counts, gene_ids, gene_names, barcodes)


def deterministic_split(
    n_cells: int,
    seed: int,
    n_fit: int,
    n_development: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_fit + n_development >= n_cells:
        raise ValueError("split must leave at least one test cell")
    order = np.random.default_rng(seed).permutation(n_cells)
    fit = np.sort(order[:n_fit])
    development = np.sort(order[n_fit : n_fit + n_development])
    test = np.sort(order[n_fit + n_development :])
    return fit, development, test


def target_free_log_normalize(
    counts: sp.csc_matrix,
    target_indices: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Normalize by molecules outside the target panel.

    Consequently changing a held-out target count cannot change any input
    feature through the library-size denominator.
    """
    total = np.asarray(counts.sum(axis=0)).ravel().astype(float)
    target_total = np.asarray(counts[target_indices].sum(axis=0)).ravel().astype(float)
    exposure = total - target_total
    if np.any(exposure <= 0.0):
        raise ValueError("target-free library size must be positive")
    normalized = (counts @ sp.diags(10000.0 / exposure)).tocsc()
    normalized.data = np.log1p(normalized.data)
    return normalized, exposure


def select_transfer_features(
    primary: LoadedDataset,
    external: LoadedDataset,
    fit: np.ndarray,
) -> list[str]:
    id_to_index = {gene_id: index for index, gene_id in enumerate(primary.gene_ids)}
    target_indices = np.array([id_to_index[item] for item in TARGET_IDS])
    normalized, _ = target_free_log_normalize(primary.counts, target_indices)
    fit_values = normalized[:, fit]
    mean = np.asarray(fit_values.mean(axis=1)).ravel()
    second = np.asarray(fit_values.power(2).mean(axis=1)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    detection = np.asarray((fit_values > 0).mean(axis=1)).ravel()
    external_ids = set(external.gene_ids)
    target_ids = set(TARGET_IDS)
    candidates = [
        index
        for index, (gene_id, gene_name) in enumerate(
            zip(primary.gene_ids, primary.gene_names)
        )
        if gene_id in external_ids
        and gene_id not in target_ids
        and not gene_name.upper().startswith("MT-")
        and 0.03 <= detection[index] <= 0.97
    ]
    ordered = sorted(candidates, key=lambda index: (variance[index], primary.gene_ids[index]))
    selected = ordered[-N_INPUT_GENES:]
    if len(selected) != N_INPUT_GENES:
        raise ValueError("not enough common candidate genes")
    return [primary.gene_ids[index] for index in selected]


def prepare_dataset(
    loaded: LoadedDataset,
    train: np.ndarray,
    test: np.ndarray,
    input_ids: Iterable[str],
) -> PreparedDataset:
    id_to_index = {gene_id: index for index, gene_id in enumerate(loaded.gene_ids)}
    missing = sorted((set(input_ids) | set(TARGET_IDS)) - set(id_to_index))
    if missing:
        raise ValueError(f"{loaded.spec.key} missing required Ensembl IDs: {missing}")
    input_ids = list(input_ids)
    input_indices = np.array([id_to_index[item] for item in input_ids])
    target_indices = np.array([id_to_index[item] for item in TARGET_IDS])
    normalized, exposure = target_free_log_normalize(loaded.counts, target_indices)
    x = normalized[input_indices].T.toarray()
    y = normalized[target_indices].T.toarray()
    x_mean = x[train].mean(axis=0)
    x_std = x[train].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y[train].mean(axis=0)
    y_std = y[train].std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    embedding = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(x[train]).transform(x)
    train_mask = np.zeros(x.shape[0], dtype=bool)
    train_mask[train] = True
    input_names = [loaded.gene_names[index] for index in input_indices]
    return PreparedDataset(
        x=x,
        y=y,
        embedding=embedding,
        train=np.asarray(train),
        test=np.asarray(test),
        train_mask=train_mask,
        input_ids=input_ids,
        input_names=input_names,
        target_names=list(TARGET_NAMES),
        target_free_library=exposure,
    )


def mse(y: np.ndarray, prediction: np.ndarray, index: np.ndarray) -> float:
    return float(np.mean((y[index] - prediction[index]) ** 2))


def mae(y: np.ndarray, prediction: np.ndarray, index: np.ndarray) -> float:
    return float(np.mean(np.abs(y[index] - prediction[index])))


def _knn_prediction(
    embedding: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    neighbors: int,
    weighting: str,
) -> np.ndarray:
    query = NearestNeighbors(n_neighbors=neighbors + 1).fit(embedding[train])
    distance, local_index = query.kneighbors(embedding)
    prediction = np.empty_like(y)
    train_values = y[train]
    for cell in range(y.shape[0]):
        keep = [
            position
            for position in range(local_index.shape[1])
            if int(train[local_index[cell, position]]) != cell
        ][:neighbors]
        if weighting == "uniform":
            weight = np.full(len(keep), 1.0 / len(keep))
        elif weighting == "distance":
            weight = 1.0 / np.maximum(distance[cell, keep], 1e-6)
            weight /= weight.sum()
        else:
            raise ValueError(f"unknown kNN weighting {weighting}")
        prediction[cell] = weight @ train_values[local_index[cell, keep]]
    return prediction


def _tree_model(family: str, params: dict[str, Any], seed: int):
    common = dict(
        n_estimators=N_TREES,
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        random_state=int(seed),
        n_jobs=1,
    )
    if family == "extra_trees":
        return ExtraTreesRegressor(**common)
    if family == "random_forest":
        return RandomForestRegressor(**common)
    raise ValueError(f"unknown tree family {family}")


def baseline_prediction(
    family: str,
    params: dict[str, Any],
    data: PreparedDataset,
    seed: int = SEED,
) -> np.ndarray:
    if family == "mean":
        return np.tile(data.y[data.train].mean(axis=0), (data.y.shape[0], 1))
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"])).fit(
            data.x[data.train], data.y[data.train]
        ).predict(data.x)
    if family == "pls":
        return PLSRegression(
            n_components=int(params["n_components"]), scale=False, max_iter=1000
        ).fit(data.x[data.train], data.y[data.train]).predict(data.x)
    if family == "knn":
        return _knn_prediction(
            data.embedding,
            data.y,
            data.train,
            int(params["neighbors"]),
            str(params["weighting"]),
        )
    if family in {"extra_trees", "random_forest"}:
        model = _tree_model(family, params, seed)
        return model.fit(data.x[data.train], data.y[data.train]).predict(data.x)
    raise ValueError(f"unknown baseline family {family}")


BASELINE_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "mean": ({},),
    "ridge": tuple({"alpha": value} for value in (0.1, 1.0, 10.0, 100.0, 1000.0)),
    "pls": tuple({"n_components": value} for value in (4, 8, 16, 32)),
    "knn": tuple(
        {"neighbors": neighbors, "weighting": weighting}
        for neighbors in (5, 10, 15, 25, 40)
        for weighting in ("uniform", "distance")
    ),
    "random_forest": tuple(
        {"min_samples_leaf": leaf, "max_features": features}
        for leaf in (1, 5, 10)
        for features in (0.5, 1.0)
    ),
    "extra_trees": tuple(
        {"min_samples_leaf": leaf, "max_features": features}
        for leaf in (1, 2, 5, 10)
        for features in (0.5, 1.0)
    ),
}


def tune_baselines(
    data: PreparedDataset,
    development: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for family, grid in BASELINE_GRIDS.items():
        family_rows = []
        for candidate_index, params in enumerate(grid):
            prediction = baseline_prediction(family, params, data, SEED + candidate_index)
            row = {
                "family": family,
                "parameters": params,
                "development_mse": mse(data.y, prediction, development),
            }
            rows.append(row)
            family_rows.append(row)
        winner = min(
            family_rows,
            key=lambda row: (
                row["development_mse"],
                json.dumps(row["parameters"], sort_keys=True),
            ),
        )
        selected[family] = dict(winner["parameters"])
    return selected, rows


def graph_transition(embedding: np.ndarray) -> tuple[sp.csr_matrix, float]:
    query = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(embedding)
    distance, neighbors = query.kneighbors(embedding)
    distance = distance[:, 1:]
    neighbors = neighbors[:, 1:]
    positive = distance[distance > 0.0]
    scale = float(np.median(positive))
    return transition_matrix(neighbors, distance, scale), scale


def cross_fitted_extra_trees(
    data: PreparedDataset,
    params: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return full-fit predictions and target-wise out-of-fold train predictions."""
    full = _tree_model("extra_trees", params, seed).fit(
        data.x[data.train], data.y[data.train]
    )
    prediction = full.predict(data.x)
    out_of_fold = np.full_like(data.y, np.nan)
    audit: list[dict[str, Any]] = []
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_local, held_local) in enumerate(splitter.split(data.train)):
        fold_train = data.train[train_local]
        fold_held = data.train[held_local]
        model = _tree_model("extra_trees", params, seed + fold + 1).fit(
            data.x[fold_train], data.y[fold_train]
        )
        out_of_fold[fold_held] = model.predict(data.x[fold_held])
        audit.append(
            {
                "fold": fold,
                "training_cells": int(len(fold_train)),
                "held_out_cells": int(len(fold_held)),
                "overlap": int(len(set(fold_train) & set(fold_held))),
                "held_out_index_sha256": hashlib.sha256(
                    np.asarray(fold_held, dtype=np.int64).tobytes()
                ).hexdigest(),
            }
        )
    if not np.isfinite(out_of_fold[data.train]).all():
        raise RuntimeError("cross-fitting did not cover every training cell")
    return prediction, out_of_fold, audit


def optimal_nonnegative_blend(error: np.ndarray, correction: np.ndarray) -> float:
    """Return the exact least-squares blend on ``[0,1]``."""
    denominator = float(np.sum(correction * correction))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    unconstrained = float(np.sum(error * correction) / denominator)
    return float(np.clip(unconstrained, 0.0, 1.0))


def tune_correction(
    data: PreparedDataset,
    development: np.ndarray,
    extra_parameters: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base, out_of_fold, fold_audit = cross_fitted_extra_trees(
        data, extra_parameters, SEED + 200
    )
    transition, scale = graph_transition(data.embedding)
    residual_seed = np.zeros_like(data.y)
    residual_seed[data.train] = data.y[data.train] - out_of_fold[data.train]
    rows: list[dict[str, Any]] = []
    for graph_weight in GRAPH_WEIGHTS:
        field, certificate = solve_residual_field_batched(
            transition,
            data.train_mask,
            residual_seed,
            REGULARIZATION,
            graph_weight,
            rtol=1e-8,
        )
        error = data.y[development] - base[development]
        blend = optimal_nonnegative_blend(error, field[development])
        prediction = base + blend * field
        rows.append(
            {
                "graph_weight": graph_weight,
                "blend": blend,
                "development_mse": mse(data.y, prediction, development),
                "base_development_mse": mse(data.y, base, development),
                "synchronized_iterations": certificate.synchronized_iterations,
                "max_relative_residual": certificate.max_relative_residual,
            }
        )
    selected = min(rows, key=lambda row: (row["development_mse"], row["graph_weight"]))
    audit = {
        "folds": fold_audit,
        "distance_scale": scale,
        "candidate_graph_weights": list(GRAPH_WEIGHTS),
        "test_targets_used": False,
    }
    return dict(selected), rows, audit


def metric_block(y: np.ndarray, predictions: dict[str, np.ndarray], test: np.ndarray) -> dict:
    return {
        name: {"mse": mse(y, prediction, test), "mae": mae(y, prediction, test)}
        for name, prediction in predictions.items()
    }


def gene_metric_rows(
    y: np.ndarray,
    base: np.ndarray,
    method: np.ndarray,
    test: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for column, gene in enumerate(TARGET_NAMES):
        base_error = float(np.mean((y[test, column] - base[test, column]) ** 2))
        method_error = float(np.mean((y[test, column] - method[test, column]) ** 2))
        statistic = spearmanr(y[test, column], method[test, column]).statistic
        rows.append(
            {
                "gene": gene,
                "extra_trees_mse": base_error,
                "method_mse": method_error,
                "method_minus_extra_trees": method_error - base_error,
                "method_spearman": float(statistic),
            }
        )
    return rows


def evaluate_frozen(
    loaded: LoadedDataset,
    train: np.ndarray,
    test: np.ndarray,
    input_ids: list[str],
    selected_baselines: dict[str, dict[str, Any]],
    selected_correction: dict[str, Any],
    seed: int,
    include_all_baselines: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = prepare_dataset(loaded, train, test, input_ids)
    predictions: dict[str, np.ndarray] = {}
    families = list(BASELINE_GRIDS) if include_all_baselines else ["extra_trees"]
    for offset, family in enumerate(families):
        predictions[family] = baseline_prediction(
            family, selected_baselines[family], data, seed + offset
        )
    base, out_of_fold, fold_audit = cross_fitted_extra_trees(
        data, selected_baselines["extra_trees"], seed + 100
    )
    # Use exactly the same full-fit ExtraTrees prediction for the baseline and
    # correction so the comparison isolates residual propagation.
    predictions["extra_trees"] = base
    residual_seed = np.zeros_like(data.y)
    residual_seed[data.train] = data.y[data.train] - out_of_fold[data.train]
    transition, distance_scale = graph_transition(data.embedding)
    field, certificate = solve_residual_field_batched(
        transition,
        data.train_mask,
        residual_seed,
        REGULARIZATION,
        float(selected_correction["graph_weight"]),
        rtol=FINAL_RTOL,
    )
    method = base + float(selected_correction["blend"]) * field
    predictions["method"] = method
    metrics = metric_block(data.y, predictions, data.test)
    result = {
        "dataset": loaded.spec.key,
        "train_cells": int(len(train)),
        "test_cells": int(len(test)),
        "metrics": metrics,
        "gene_metrics": gene_metric_rows(data.y, base, method, data.test),
        "operator": {
            "shape": [int(value) for value in transition.shape],
            "nonzeros": int(transition.nnz),
            "distance_scale": distance_scale,
            "sparse_storage_scalars": sparse_storage_scalars(transition),
            "dense_storage_scalars": int(transition.shape[0] ** 2),
            "storage_fraction": sparse_storage_scalars(transition)
            / float(transition.shape[0] ** 2),
            "dense_operator_formed_in_recovery": False,
        },
        "solver": {
            "all_converged": certificate.converged,
            "synchronized_iterations": certificate.synchronized_iterations,
            "sparse_operator_launches": certificate.sparse_operator_launches,
            "max_relative_residual": certificate.max_relative_residual,
            "per_target": [
                {"gene": gene, **asdict(item)}
                for gene, item in zip(TARGET_NAMES, certificate.per_column)
            ],
        },
        "cross_fitting": fold_audit,
        "target_free_library": {
            "minimum": float(data.target_free_library.min()),
            "median": float(np.median(data.target_free_library)),
            "maximum": float(data.target_free_library.max()),
        },
        "input_names": data.input_names,
    }
    context = {
        "data": data,
        "transition": transition,
        "residual_seed": residual_seed,
        "base": base,
        "method": method,
        "field": field,
    }
    return result, context


def dense_equivalence_audit(context: dict[str, Any]) -> list[dict[str, Any]]:
    data: PreparedDataset = context["data"]
    rows = []
    for n_cells in (64, 128, 256):
        embedding = data.embedding[:n_cells]
        query = NearestNeighbors(n_neighbors=min(N_NEIGHBORS + 1, n_cells)).fit(embedding)
        distance, neighbors = query.kneighbors(embedding)
        distance, neighbors = distance[:, 1:], neighbors[:, 1:]
        scale = float(np.median(distance[distance > 0.0]))
        transition = transition_matrix(neighbors, distance, scale)
        mask = data.train_mask[:n_cells]
        seed = context["residual_seed"][:n_cells, :1]
        matrix_free, certificate = solve_residual_field_batched(
            transition,
            mask,
            seed,
            REGULARIZATION,
            3.0,
            rtol=1e-10,
        )
        dense = explicit_dense_normal(transition, mask, REGULARIZATION, 3.0)
        right = mask[:, None] * seed
        direct = np.linalg.solve(dense, right)
        rows.append(
            {
                "n_cells": n_cells,
                "relative_solution_difference": float(
                    np.linalg.norm(matrix_free - direct)
                    / max(np.linalg.norm(direct), np.finfo(float).tiny)
                ),
                "relative_residual": certificate.max_relative_residual,
            }
        )
    return rows


def solver_runtime_audit(
    context: dict[str, Any],
    graph_weight: float,
    repeats: int = 7,
) -> list[dict[str, Any]]:
    transition = context["transition"]
    data: PreparedDataset = context["data"]
    residual_seed = context["residual_seed"]
    rows = []
    for right_sides in (1, 2, 4, 8, 12):
        residual = residual_seed[:, :right_sides]
        # Warm both paths before measuring.
        solve_residual_field_batched(
            transition, data.train_mask, residual, REGULARIZATION, graph_weight
        )
        solve_residual_field_scalar(
            transition, data.train_mask, residual, REGULARIZATION, graph_weight
        )
        batched_times: list[float] = []
        scalar_times: list[float] = []
        agreement = 0.0
        batch_certificate = None
        scalar_certificate = None
        for repeat in range(repeats):
            order = ("batched", "scalar") if repeat % 2 == 0 else ("scalar", "batched")
            solutions: dict[str, np.ndarray] = {}
            for method in order:
                started = time.perf_counter()
                if method == "batched":
                    solution, batch_certificate = solve_residual_field_batched(
                        transition,
                        data.train_mask,
                        residual,
                        REGULARIZATION,
                        graph_weight,
                    )
                    batched_times.append(time.perf_counter() - started)
                else:
                    solution, scalar_certificate = solve_residual_field_scalar(
                        transition,
                        data.train_mask,
                        residual,
                        REGULARIZATION,
                        graph_weight,
                    )
                    scalar_times.append(time.perf_counter() - started)
                solutions[method] = solution
            agreement = max(
                agreement,
                float(
                    np.linalg.norm(solutions["batched"] - solutions["scalar"])
                    / max(np.linalg.norm(solutions["scalar"]), np.finfo(float).tiny)
                ),
            )
        assert batch_certificate is not None and scalar_certificate is not None
        rows.append(
            {
                "right_hand_sides": right_sides,
                "batched_median_seconds": statistics.median(batched_times),
                "batched_q1_seconds": float(np.quantile(batched_times, 0.25)),
                "batched_q3_seconds": float(np.quantile(batched_times, 0.75)),
                "scalar_median_seconds": statistics.median(scalar_times),
                "scalar_q1_seconds": float(np.quantile(scalar_times, 0.25)),
                "scalar_q3_seconds": float(np.quantile(scalar_times, 0.75)),
                "scalar_over_batched": statistics.median(scalar_times)
                / statistics.median(batched_times),
                "relative_solution_difference": agreement,
                "batched_operator_launches": batch_certificate.sparse_operator_launches,
                "scalar_operator_launches": int(
                    sum(item.iterations + 1 for item in scalar_certificate)
                ),
            }
        )
    return rows


def repeated_split_audit(
    primary: LoadedDataset,
    input_ids: list[str],
    selected_baselines: dict[str, dict[str, Any]],
    selected_correction: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for offset, seed in enumerate(range(SEED + 2, SEED + 7)):
        train, _, test = deterministic_split(primary.counts.shape[1], seed, 2160, 0)
        result, _ = evaluate_frozen(
            primary,
            train,
            test,
            input_ids,
            selected_baselines,
            selected_correction,
            seed=seed + 1000,
            include_all_baselines=False,
        )
        base = result["metrics"]["extra_trees"]["mse"]
        method = result["metrics"]["method"]["mse"]
        rows.append(
            {
                "split_seed": seed,
                "extra_trees_mse": base,
                "method_mse": method,
                "method_minus_extra_trees": method - base,
                "relative_change": (method - base) / base,
                "gene_wins": sum(
                    item["method_minus_extra_trees"] < 0.0
                    for item in result["gene_metrics"]
                ),
            }
        )
    return rows


def _dataset_audit(loaded: LoadedDataset) -> dict[str, Any]:
    return {
        **asdict(loaded.spec),
        "archive_bytes": int(
            (package_root() / "data" / "source" / loaded.spec.archive).stat().st_size
        ),
        "cells": loaded.counts.shape[1],
        "genes": loaded.counts.shape[0],
        "barcode_count": len(loaded.barcodes),
        "gene_id_count": len(loaded.gene_ids),
    }


def _provenance_scalar(value: Any) -> str | int | float | bool | None:
    """Convert optional environment probes to stable JSON scalar values.

    ``threadpoolctl`` can expose a callable in its optional ``version`` field
    for SciPy's bundled BLAS on macOS.  A qualified callable name is stable;
    ``repr(value)`` is not, because it embeds a process-specific address.
    Scientific result fields never pass through this conversion.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _provenance_scalar(value.item())
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"
    return str(value)


def _environment_metadata() -> dict[str, Any]:
    """Return deterministic, JSON-safe runtime provenance."""
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "reference_distributions": {
            name: importlib.metadata.version(name)
            for name in REFERENCE_DISTRIBUTIONS
        },
        "numpy_blas": _provenance_scalar(
            np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
        ),
        "threadpools": [
            {
                key: _provenance_scalar(pool.get(key))
                for key in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "version",
                )
            }
            for pool in threadpool_info()
        ],
        "thread_environment": {
            name: os.environ[name] for name in THREAD_ENVIRONMENT
        },
    }
    # This intentionally has no ``default=`` fallback: a future unsupported
    # provenance type should fail before the expensive scientific computation.
    json.dumps(metadata, sort_keys=True)
    return metadata


def _assert_json_tree(value: Any, path: str = "root") -> None:
    """Reject unsupported result types with their exact scientific path."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON key at {path}: {key!r}")
            _assert_json_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_json_tree(child, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def run_experiment(root: Optional[Path] = None, write_artifacts: bool = False) -> dict:
    root = Path(root) if root is not None else package_root()
    # Fail immediately on provenance-schema defects, before model fitting.
    _environment_metadata()
    primary = load_dataset(root, "pbmc3k")
    external = load_dataset(root, "pbmc4k")

    fit, development, test = deterministic_split(2700, SEED, 1620, 540)
    input_ids = select_transfer_features(primary, external, fit)
    tuning_data = prepare_dataset(primary, fit, development, input_ids)
    selected_baselines, baseline_tuning = tune_baselines(tuning_data, development)
    selected_correction, correction_tuning, tuning_audit = tune_correction(
        tuning_data, development, selected_baselines["extra_trees"]
    )

    primary_train = np.sort(np.concatenate((fit, development)))
    primary_result, primary_context = evaluate_frozen(
        primary,
        primary_train,
        test,
        input_ids,
        selected_baselines,
        selected_correction,
        seed=SEED + 500,
    )
    external_train, _, external_test = deterministic_split(4340, SEED + 1, 3472, 0)
    external_result, _ = evaluate_frozen(
        external,
        external_train,
        external_test,
        input_ids,
        selected_baselines,
        selected_correction,
        seed=SEED + 600,
    )
    repeats = repeated_split_audit(
        primary, input_ids, selected_baselines, selected_correction
    )
    dense_audit = dense_equivalence_audit(primary_context)
    runtime = solver_runtime_audit(
        primary_context, float(selected_correction["graph_weight"])
    )

    id_to_primary = {item: index for index, item in enumerate(primary.gene_ids)}
    id_to_external = {item: index for index, item in enumerate(external.gene_ids)}
    feature_map = [
        {
            "ensembl_id": item,
            "pbmc3k_name": primary.gene_names[id_to_primary[item]],
            "pbmc4k_name": external.gene_names[id_to_external[item]],
        }
        for item in input_ids
    ]

    results = {
        "schema_version": 2,
        "title": "Leakage-Audited Matrix-Free Residual Propagation for Gene Measurements",
        "seed": SEED,
        "data": {key: _dataset_audit(value) for key, value in (("pbmc3k", primary), ("pbmc4k", external))},
        "protocol": {
            "target_ids": list(TARGET_IDS),
            "target_genes": list(TARGET_NAMES),
            "input_gene_ids": input_ids,
            "input_gene_name_map": feature_map,
            "input_genes": N_INPUT_GENES,
            "pca_components": N_COMPONENTS,
            "graph_neighbors": N_NEIGHBORS,
            "regularization": REGULARIZATION,
            "cross_folds": N_FOLDS,
            "tree_estimators": N_TREES,
            "primary_fit_cells": len(fit),
            "primary_development_cells": len(development),
            "primary_final_train_cells": len(primary_train),
            "primary_test_cells": len(test),
            "external_train_cells": len(external_train),
            "external_test_cells": len(external_test),
            "tree_fit_threads": 1,
            "reference_environment": {
                "python": "CPython 3.13.12",
                "system": "macOS 26.6",
                "machine": "arm64",
                "numerical_distribution": "version-locked PyPI wheels",
                "numpy_blas_lapack": "Accelerate",
                "lock_file": "code/requirements-lock.txt",
                "lock_sha256": REFERENCE_LOCK_SHA256,
                "hash_locked": False,
            },
            "normalization_denominator_excludes_all_target_counts": True,
            "graph_uses_test_covariates": True,
            "test_targets_used_for_selection": False,
            "selection_is_retrospective_not_preregistered": True,
        },
        "selected_baselines": selected_baselines,
        "selected_correction": {
            "graph_weight": selected_correction["graph_weight"],
            "blend": selected_correction["blend"],
            "development_mse": selected_correction["development_mse"],
            "base_development_mse": selected_correction["base_development_mse"],
        },
        "baseline_tuning": baseline_tuning,
        "correction_tuning": correction_tuning,
        "tuning_audit": tuning_audit,
        "primary": primary_result,
        "external": external_result,
        "repeated_splits": repeats,
        "dense_equivalence": dense_audit,
        "solver_runtime": runtime,
        "environment": _environment_metadata(),
        "claim_boundaries": {
            "worldwide_algorithmic_novelty_claimed": False,
            "biological_recovery_claimed": False,
            "donor_generalization_claimed": False,
            "universal_baseline_superiority_claimed": False,
            "end_to_end_speedup_claimed": False,
            "test_cells_independent_donors": False,
            "numerical_tests_are_mathematical_proofs": False,
        },
        "verification_contract": {
            "discrete_fields": "exact",
            "floating_relative_tolerance": LOCKED_RESULT_REL_TOL,
            "floating_absolute_tolerance": LOCKED_RESULT_ABS_TOL,
            "excluded_fields": [
                "environment",
                "solver_runtime",
            ],
            "method_spearman_status": "locked_in_reference_environment",
            "hypothesis_gates": [
                "primary method MSE < ExtraTrees MSE",
                "external method MSE < ExtraTrees MSE",
                "all five frozen repeated-split MSE differences < 0",
                "all PCG relative residuals < 2e-9",
                "dense relative solution differences < 1e-7",
            ],
        },
    }
    _assert_json_tree(results)
    json.dumps(results, sort_keys=True)
    if write_artifacts:
        write_release_artifacts(results, root)
    return results


def _scientific_projection(results: dict) -> dict:
    projected = json.loads(json.dumps(results))
    projected.pop("environment", None)
    projected.pop("solver_runtime", None)
    return projected


def _compare(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"key mismatch at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        if not np.isclose(left, right, rtol=LOCKED_RESULT_REL_TOL, atol=LOCKED_RESULT_ABS_TOL):
            raise AssertionError(f"value mismatch at {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def verify_locked_results(root: Optional[Path] = None) -> dict:
    root = Path(root) if root is not None else package_root()
    blas_name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    installed = {
        name: importlib.metadata.version(name) for name in REFERENCE_DISTRIBUTIONS
    }
    pools = threadpool_info()
    numerical_pools = [
        pool for pool in pools if pool.get("user_api") in {"blas", "openmp"}
    ]
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:3] != REFERENCE_PYTHON
        or platform.system() != REFERENCE_SYSTEM
        or platform.machine() != REFERENCE_MACHINE
        or platform.mac_ver()[0] != REFERENCE_MACOS
        or str(blas_name).lower() != REFERENCE_BLAS
        or installed != REFERENCE_DISTRIBUTIONS
        or not numerical_pools
        or any(pool.get("num_threads") != 1 for pool in numerical_pools)
        or any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT)
    ):
        raise RuntimeError(
            "locked verification requires CPython 3.13.12 on macOS 26.6 arm64 with "
            "the exact version lock, PyPI NumPy using Accelerate, and one-thread pools; "
            "install code/requirements-lock.txt before importing mfcell"
        )
    locked = json.loads((root / "results" / "results.json").read_text())
    fresh = run_experiment(root=root, write_artifacts=False)
    _compare(_scientific_projection(locked), _scientific_projection(fresh))
    protocol = fresh["protocol"]
    if not protocol["normalization_denominator_excludes_all_target_counts"]:
        raise AssertionError("target counts entered the normalization denominator")
    if protocol["test_targets_used_for_selection"]:
        raise AssertionError("test targets entered model selection")
    for key, specification in DATASETS.items():
        if fresh["data"][key]["sha256"] != specification.sha256:
            raise AssertionError(f"source digest mismatch for {key}")
    for section in ("primary", "external"):
        block = fresh[section]
        if block["metrics"]["method"]["mse"] >= block["metrics"]["extra_trees"]["mse"]:
            raise AssertionError(f"correction no longer beats ExtraTrees on {section}")
        if not block["solver"]["all_converged"]:
            raise AssertionError(f"PCG did not converge on {section}")
        if block["solver"]["max_relative_residual"] >= 2e-9:
            raise AssertionError(f"PCG residual gate failed on {section}")
        if any(fold["overlap"] for fold in block["cross_fitting"]):
            raise AssertionError(f"cross-fitting overlap on {section}")
    if any(row["method_minus_extra_trees"] >= 0 for row in fresh["repeated_splits"]):
        raise AssertionError("a frozen repeated split no longer favors correction")
    if max(row["relative_solution_difference"] for row in fresh["dense_equivalence"]) >= 1e-7:
        raise AssertionError("dense-equivalence gate failed")
    if any(fresh["claim_boundaries"].values()):
        raise AssertionError("a prohibited broad claim is enabled")
    return fresh


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def create_figures(results: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = ["Mean", "Ridge", "PLS", "kNN", "RF", "ExtraTrees", "CF-MFRP"]
    keys = ["mean", "ridge", "pls", "knn", "random_forest", "extra_trees", "method"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for axis, section, title in zip(
        axes, (results["primary"], results["external"]), ("PBMC3k test", "PBMC4k transfer test")
    ):
        values = [section["metrics"][key]["mse"] for key in keys]
        axis.plot(range(len(values)), values, "o-", color="#2F6B9A", lw=1.4)
        axis.set_xticks(range(len(values)), labels, rotation=45, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Standardized test MSE")
    fig.tight_layout()
    _save_pdf(fig, output / "baseline_comparison.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axis, section, title in zip(
        axes, (results["primary"], results["external"]), ("PBMC3k", "PBMC4k")
    ):
        rows = section["gene_metrics"]
        delta = [row["method_minus_extra_trees"] for row in rows]
        axis.axhline(0.0, color="black", lw=0.8)
        axis.plot(range(len(rows)), delta, "o-", color="#B04A5A", lw=1.2)
        axis.set_xticks(range(len(rows)), [row["gene"] for row in rows], rotation=60, ha="right", fontsize=7)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Method minus ExtraTrees MSE")
    fig.tight_layout()
    _save_pdf(fig, output / "gene_differences.pdf")

    rows = results["correction_tuning"]
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    weights = [row["graph_weight"] for row in rows]
    axis.plot(weights, [row["development_mse"] for row in rows], "o-", label="corrected")
    axis.axhline(rows[0]["base_development_mse"], color="#777777", ls="--", label="ExtraTrees")
    axis.set_xscale("log")
    axis.set_xlabel(r"Graph weight $\lambda$")
    axis.set_ylabel("PBMC3k development MSE")
    axis.grid(alpha=0.25)
    second = axis.twinx()
    second.plot(weights, [row["blend"] for row in rows], "s:", color="#D07C28", label=r"$\alpha^*$")
    second.set_ylabel(r"Analytic blend $\alpha^*$")
    handles, labels_left = axis.get_legend_handles_labels()
    handles2, labels_right = second.get_legend_handles_labels()
    axis.legend(handles + handles2, labels_left + labels_right, frameon=False, fontsize=8)
    fig.tight_layout()
    _save_pdf(fig, output / "regularization_path.pdf")

    repeated = results["repeated_splits"]
    fig, axis = plt.subplots(figsize=(5.8, 3.1))
    axis.axhline(0.0, color="black", lw=0.8)
    axis.plot(
        [row["split_seed"] - SEED for row in repeated],
        [100.0 * row["relative_change"] for row in repeated],
        "o-",
        color="#4E8B57",
    )
    axis.set_xlabel("Prespecified split-seed offset")
    axis.set_ylabel("MSE change vs ExtraTrees (percent)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save_pdf(fig, output / "split_stability.pdf")

    runtime = results["solver_runtime"]
    right_sides = [row["right_hand_sides"] for row in runtime]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for prefix, label, color in (
        ("scalar", "scalar PCG loop", "#888888"),
        ("batched", "synchronized PCG", "#3B75AF"),
    ):
        median = np.array([row[f"{prefix}_median_seconds"] for row in runtime])
        low = np.array([row[f"{prefix}_q1_seconds"] for row in runtime])
        high = np.array([row[f"{prefix}_q3_seconds"] for row in runtime])
        axes[0].plot(right_sides, median, "o-", label=label, color=color)
        axes[0].fill_between(right_sides, low, high, alpha=0.18, color=color)
    axes[0].set_xlabel("Right-hand sides")
    axes[0].set_ylabel("Median solve seconds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(right_sides, [row["scalar_operator_launches"] for row in runtime], "o-", label="scalar")
    axes[1].plot(right_sides, [row["batched_operator_launches"] for row in runtime], "s-", label="synchronized")
    axes[1].set_xlabel("Right-hand sides")
    axes[1].set_ylabel("Sparse operator launches")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_pdf(fig, output / "solver_scaling.pdf")


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(results: dict, result_output: Path, table_output: Path) -> None:
    result_output.mkdir(parents=True, exist_ok=True)
    table_output.mkdir(parents=True, exist_ok=True)
    primary = results["primary"]
    external = results["external"]
    pm = primary["metrics"]
    em = external["metrics"]
    correction = results["selected_correction"]
    runtime = results["solver_runtime"][-1]
    macros = [
        rf"\newcommand{{\PrimaryBaseMSE}}{{{pm['extra_trees']['mse']:.6f}}}",
        rf"\newcommand{{\PrimaryMethodMSE}}{{{pm['method']['mse']:.6f}}}",
        rf"\newcommand{{\ExternalBaseMSE}}{{{em['extra_trees']['mse']:.6f}}}",
        rf"\newcommand{{\ExternalMethodMSE}}{{{em['method']['mse']:.6f}}}",
        rf"\newcommand{{\PrimaryRelativeGain}}{{{100*(pm['extra_trees']['mse']-pm['method']['mse'])/pm['extra_trees']['mse']:.2f}\%}}",
        rf"\newcommand{{\ExternalRelativeGain}}{{{100*(em['extra_trees']['mse']-em['method']['mse'])/em['extra_trees']['mse']:.2f}\%}}",
        rf"\newcommand{{\SelectedLambda}}{{{correction['graph_weight']:.2g}}}",
        rf"\newcommand{{\SelectedBlend}}{{{correction['blend']:.3f}}}",
        rf"\newcommand{{\PrimaryResidual}}{{{primary['solver']['max_relative_residual']:.2e}}}",
        rf"\newcommand{{\ExternalResidual}}{{{external['solver']['max_relative_residual']:.2e}}}",
        rf"\newcommand{{\PrimaryStorage}}{{{100*primary['operator']['storage_fraction']:.3f}\%}}",
        rf"\newcommand{{\ExternalStorage}}{{{100*external['operator']['storage_fraction']:.3f}\%}}",
        rf"\newcommand{{\BatchSpeedRatio}}{{{runtime['scalar_over_batched']:.2f}}}",
        rf"\newcommand{{\BatchAgreement}}{{{runtime['relative_solution_difference']:.2e}}}",
        rf"\newcommand{{\RepeatWins}}{{{sum(row['method_minus_extra_trees'] < 0 for row in results['repeated_splits'])}}}",
    ]
    (result_output / "results.tex").write_text("\n".join(macros) + "\n")

    data_lines = [
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        r"Source & Genes & Cells & Nonzeros & Molecules & Bytes & SHA-256 prefix\\", r"\midrule",
    ]
    for key in ("pbmc3k", "pbmc4k"):
        row = results["data"][key]
        data_lines.append(
            f"{row['title']} & {row['shape'][0]:,} & {row['shape'][1]:,} & {row['nonzeros']:,} & "
            f"{row['total_counts']:,} & {row['archive_bytes']:,} & \\texttt{{{row['sha256'][:12]}}}\\\\"
        )
    data_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_output / "datasets.tex").write_text("\n".join(data_lines) + "\n")

    baseline_lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & PBMC3k MSE & PBMC3k MAE & PBMC4k MSE & PBMC4k MAE\\", r"\midrule",
    ]
    display = {
        "mean": "Mean", "ridge": "Ridge", "pls": "PLS", "knn": "$k$NN",
        "random_forest": "Random forest", "extra_trees": "ExtraTrees", "method": "CF-MFRP",
    }
    for key in ("mean", "ridge", "pls", "knn", "random_forest", "extra_trees", "method"):
        baseline_lines.append(
            f"{display[key]} & {pm[key]['mse']:.4f} & {pm[key]['mae']:.4f} & "
            f"{em[key]['mse']:.4f} & {em[key]['mae']:.4f}\\\\"
        )
    baseline_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_output / "baselines.tex").write_text("\n".join(baseline_lines) + "\n")

    numerical_lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Source & $n$ & nnz$(P)$ & CSR/dense & Iterations & Max residual\\", r"\midrule",
    ]
    for section in (primary, external):
        numerical_lines.append(
            f"{section['dataset']} & {section['operator']['shape'][0]} & {section['operator']['nonzeros']} & "
            f"{section['operator']['storage_fraction']:.4f} & {section['solver']['synchronized_iterations']} & "
            f"{section['solver']['max_relative_residual']:.2e}\\\\"
        )
    numerical_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_output / "numerics.tex").write_text("\n".join(numerical_lines) + "\n")

    gene_lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Gene & 3k ET & 3k CF & $\Delta$3k & 4k ET & 4k CF & $\Delta$4k\\", r"\midrule",
    ]
    for left, right in zip(primary["gene_metrics"], external["gene_metrics"]):
        gene_lines.append(
            f"{_tex_escape(left['gene'])} & {left['extra_trees_mse']:.4f} & {left['method_mse']:.4f} & "
            f"{left['method_minus_extra_trees']:+.4f} & {right['extra_trees_mse']:.4f} & "
            f"{right['method_mse']:.4f} & {right['method_minus_extra_trees']:+.4f}\\\\"
        )
    gene_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_output / "gene_metrics.tex").write_text("\n".join(gene_lines) + "\n")


def write_release_artifacts(results: dict, root: Path) -> None:
    result_dir = root / "results"
    asset_dir = root / "manuscript_assets"
    figure_dir = asset_dir / "figures"
    table_dir = asset_dir / "tables"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    write_tex(results, result_dir, table_dir)
    create_figures(results, figure_dir)
    package_results = root / "src" / "mfcell" / "results"
    if package_results.parent.is_dir():
        package_results.mkdir(parents=True, exist_ok=True)
        (package_results / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
        write_tex(results, package_results, table_dir)
