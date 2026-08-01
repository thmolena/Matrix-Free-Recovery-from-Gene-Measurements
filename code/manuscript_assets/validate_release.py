#!/usr/bin/env python3
"""Validate the leakage-audited manuscript and offline reproduction package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ASSETS = Path(__file__).resolve().parent
REPO = ASSETS.parents[1]
MANIFEST = ASSETS / "evidence_manifest.json"

DATA_DIGESTS = {
    "pbmc3k": "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5",
    "pbmc4k": "097ff873bae645bec5586898ac002ccebbce952d52ac19ea0a35c27c4820df26",
}
DATA_FILES = {
    "pbmc3k": "pbmc3k_filtered_gene_bc_matrices.tar.gz",
    "pbmc4k": "pbmc4k_filtered_gene_bc_matrices.tar.gz",
}
FIGURES = {
    "code/manuscript_assets/figures/baseline_comparison.pdf",
    "code/manuscript_assets/figures/gene_differences.pdf",
    "code/manuscript_assets/figures/regularization_path.pdf",
    "code/manuscript_assets/figures/solver_scaling.pdf",
    "code/manuscript_assets/figures/split_stability.pdf",
}
TABLE_INPUTS = {
    "code/manuscript_assets/tables/baselines.tex",
    "code/manuscript_assets/tables/datasets.tex",
    "code/manuscript_assets/tables/gene_metrics.tex",
    "code/manuscript_assets/tables/numerics.tex",
}
MANIFEST_PATHS = sorted(
    {
        "main.tex",
        "main.pdf",
        "README.md",
        "code/README.md",
        "code/pyproject.toml",
        "code/requirements-lock.txt",
        "code/requirements-test-lock.txt",
        "code/data/README.md",
        "code/data/THIRD_PARTY.md",
        "code/data/checksums.sha256",
        "code/manuscript_assets/main.bbl",
        "code/manuscript_assets/references.bib",
        "code/manuscript_assets/figures/baseline_comparison.png",
        "code/manuscript_assets/validate_release.py",
        "code/results/results.json",
        "code/results/results.tex",
        "code/results/replay_env_b.json",
        "code/results/replay_manifest.json",
        "code/scripts/download_data.py",
        "code/scripts/make_figures.py",
        "code/scripts/reproduce.py",
        "code/scripts/validate_release.py",
        "code/src/mfcell/__init__.py",
        "code/src/mfcell/experiment.py",
        "code/src/mfcell/matrix_free.py",
        "code/src/mfcell/reproduce.py",
        "code/tests/conftest.py",
        "code/tests/test_reproduction.py",
        *FIGURES,
        *TABLE_INPUTS,
        *(
            f"code/data/source/{name}"
            for name in DATA_FILES.values()
        ),
        *(
            f"code/src/mfcell/data/source/{name}"
            for name in DATA_FILES.values()
        ),
        "code/src/mfcell/results/results.json",
        "code/src/mfcell/results/results.tex",
    }
)
FORBIDDEN_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".blg",
    ".log",
    ".out",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
    ".pyc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    check(match is not None, "pdfinfo did not report a page count")
    return int(match.group(1))


def cited_keys(tex: str) -> set[str]:
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex, flags=re.DOTALL):
        cited.update(key.strip() for key in group.replace("\n", "").split(","))
    return cited


def manuscript_quantities(tex: str, bib: str) -> dict[str, int]:
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    cited = cited_keys(tex)
    check(bib_keys == cited, "bibliography contains uncited or unknown keys")
    theorem_like = sum(
        len(re.findall(rf"\\begin\{{{name}\}}", tex))
        for name in ("theorem", "proposition", "corollary")
    )
    return {
        "pages": pdf_pages(REPO / "main.pdf"),
        "bibliography_entries": len(bib_keys),
        "cited_entries": len(cited),
        "figures": len(re.findall(r"\\begin\{figure\*?\}", tex)),
        "numeric_tables": len(re.findall(r"\\begin\{table\*?\}", tex)),
        "theorem_like_statements": theorem_like,
        "proofs": len(re.findall(r"\\begin\{proof\}", tex)),
    }


def build_manifest() -> dict[str, object]:
    tex = (REPO / "main.tex").read_text(encoding="utf-8")
    bib = (ASSETS / "references.bib").read_text(encoding="utf-8")
    for relative in MANIFEST_PATHS:
        check((REPO / relative).is_file(), f"missing release artifact: {relative}")
    results = json.loads((REPO / "code/results/results.json").read_text())
    return {
        "schema_version": 3,
        "release": {
            "title": results["title"],
            "package": "mfcell",
            "package_version": "2.0.0",
            "date": "2026-07-31",
        },
        "data": {
            key: {
                field: results["data"][key][field]
                for field in (
                    "title",
                    "landing_page",
                    "url",
                    "license",
                    "sha256",
                    "archive_bytes",
                    "genes",
                    "cells",
                    "nonzeros",
                    "total_counts",
                )
            }
            for key in DATA_DIGESTS
        },
        "manuscript": manuscript_quantities(tex, bib),
        "scientific_result_sha256": sha256(REPO / "code/results/results.json"),
        "artifacts": {
            relative: sha256(REPO / relative) for relative in MANIFEST_PATHS
        },
        "claim_boundaries": results["claim_boundaries"],
    }


def validate_manifest(manifest: dict[str, object]) -> None:
    check(manifest["schema_version"] == 3, "manifest schema drift")
    check(set(manifest["artifacts"]) == set(MANIFEST_PATHS), "manifest path set drift")
    for relative, expected in manifest["artifacts"].items():
        path = REPO / relative
        check(path.is_file(), f"missing manifest artifact: {relative}")
        check(sha256(path) == expected, f"hash mismatch: {relative}")


def validate_data(results: dict[str, object], manifest: dict[str, object]) -> None:
    for key, expected in DATA_DIGESTS.items():
        name = DATA_FILES[key]
        source = REPO / "code/data/source" / name
        mirror = REPO / "code/src/mfcell/data/source" / name
        check(sha256(source) == expected, f"{key} source digest mismatch")
        check(sha256(mirror) == expected, f"{key} wheel-mirror digest mismatch")
        check(results["data"][key]["sha256"] == expected, f"{key} result digest drift")
        check(manifest["data"][key]["sha256"] == expected, f"{key} manifest digest drift")
        check(results["data"][key]["license"] == "CC BY 4.0", f"{key} license drift")
    check(
        sha256(REPO / "code/results/results.json")
        == sha256(REPO / "code/src/mfcell/results/results.json"),
        "wheel result JSON mirror drift",
    )
    check(
        sha256(REPO / "code/results/results.tex")
        == sha256(REPO / "code/src/mfcell/results/results.tex"),
        "wheel result TeX mirror drift",
    )


def validate_manuscript(manifest: dict[str, object]) -> dict[str, int]:
    tex = (REPO / "main.tex").read_text(encoding="utf-8")
    bib = (ASSETS / "references.bib").read_text(encoding="utf-8")
    normalized_tex = re.sub(r"\s+", " ", tex.lower())
    check(
        tex.startswith(r"\documentclass[10pt,letterpaper,twoside]{article}"),
        "manuscript is not a one-column research preprint",
    )
    check(
        r"\usepackage{code/manuscript_assets/researchpreprint}" in tex,
        "shared preprint style is missing",
    )

    quantities = manuscript_quantities(tex, bib)
    check(7 <= quantities["pages"] <= 20, "unexpected manuscript page count")
    check(quantities == manifest["manuscript"], "manuscript quantities drift")
    check(quantities["bibliography_entries"] >= 20, "literature screen is too small")
    check(quantities["figures"] == 5, "figure count drift")
    check(quantities["numeric_tables"] == 4, "table count drift")
    check(quantities["theorem_like_statements"] == 7, "theorem count drift")
    check(quantities["proofs"] == 7, "every theorem-like statement needs a proof")

    figures = set(re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex))
    table_inputs = {
        item
        for item in re.findall(r"\\input\{([^}]+)\}", tex)
        if "/tables/" in item
    }
    check(figures == FIGURES, "figure set drift")
    check(table_inputs == TABLE_INPUTS, "table input set drift")
    check("numerical checks support implementation correctness but are not proofs" in normalized_tex,
          "proof/experiment boundary missing")
    check("we do not call their combination the first of its kind" in normalized_tex,
          "priority boundary missing")
    check("no universal or end-to-end speedup is claimed" in (REPO / "README.md").read_text().lower(),
          "speedup boundary missing from README")
    for rejected in (
        "method_comparison_bar.pdf",
        "ablation_bar.pdf",
        "gene_error_line.pdf",
        "scaling_line.pdf",
        "topology-regularized residual recovery",
    ):
        check(rejected not in tex, f"rejected manuscript artifact remains: {rejected}")
    return quantities


def validate_results(results: dict[str, object], manifest: dict[str, object]) -> None:
    check(results["schema_version"] == 2, "result schema drift")
    check(sha256(REPO / "code/results/results.json") == manifest["scientific_result_sha256"],
          "scientific result digest drift")
    protocol = results["protocol"]
    contract = results["verification_contract"]
    check(contract["discrete_fields"] == "exact", "discrete replay contract drift")
    check(contract["floating_relative_tolerance"] == 5e-3,
          "relative replay tolerance drift")
    check(contract["floating_absolute_tolerance"] == 5e-5,
          "absolute replay tolerance drift")
    check(contract["method_spearman_status"] == "locked_in_reference_environment",
          "Spearman diagnostic status drift")
    reference = protocol["reference_environment"]
    check(reference["python"] == "CPython 3.13.12", "reference Python drift")
    check(reference["system"] == "macOS 26.6" and reference["machine"] == "arm64",
          "reference platform drift")
    check(reference["numerical_distribution"] == "version-locked PyPI wheels",
          "reference distribution drift")
    check(reference["numpy_blas_lapack"] == "Accelerate", "reference BLAS drift")
    check(reference["hash_locked"] is False, "lock disclosure drift")
    check(reference["lock_sha256"] == sha256(REPO / "code/requirements-lock.txt"),
          "reference lock digest drift")
    check(contract["excluded_fields"] == ["environment", "solver_runtime"],
          "reported metric excluded from locked projection")
    check(protocol["normalization_denominator_excludes_all_target_counts"],
          "target counts leak through normalization")
    check(not protocol["test_targets_used_for_selection"], "test targets used for selection")
    check(len(protocol["input_gene_ids"]) == 96, "input panel size drift")
    check(len(protocol["target_ids"]) == 12, "target panel size drift")
    check(not (set(protocol["input_gene_ids"]) & set(protocol["target_ids"])),
          "input and target panels overlap")
    check(results["tuning_audit"]["test_targets_used"] is False,
          "tuning audit reports test-target use")
    check(all(fold["overlap"] == 0 for fold in results["tuning_audit"]["folds"]),
          "tuning cross-fit overlap")

    for section in ("primary", "external"):
        block = results[section]
        metrics = block["metrics"]
        baseline_names = set(metrics) - {"method"}
        check(metrics["extra_trees"]["mse"] == min(metrics[name]["mse"] for name in baseline_names),
              f"ExtraTrees is not strongest {section} baseline")
        check(metrics["method"]["mse"] < metrics["extra_trees"]["mse"],
              f"correction does not beat ExtraTrees on {section}")
        check(block["solver"]["all_converged"], f"{section} PCG convergence gate failed")
        check(block["solver"]["max_relative_residual"] < 2e-9,
              f"{section} residual gate failed")
        check(block["operator"]["dense_operator_formed_in_recovery"] is False,
              f"dense {section} recovery operator reported")
        check(block["operator"]["nonzeros"] == 15 * block["operator"]["shape"][0],
              f"{section} graph sparsity drift")
        check(all(fold["overlap"] == 0 for fold in block["cross_fitting"]),
              f"{section} cross-fit overlap")
        check(min(block["target_free_library"].values()) > 0,
              f"{section} target-free exposure is not positive")
    check(sum(row["method_minus_extra_trees"] < 0 for row in results["primary"]["gene_metrics"]) == 9,
          "primary gene-win count drift")
    check(sum(row["method_minus_extra_trees"] < 0 for row in results["external"]["gene_metrics"]) == 11,
          "external gene-win count drift")

    check(len(results["repeated_splits"]) == 5, "repeat count drift")
    check(all(row["method_minus_extra_trees"] < 0 for row in results["repeated_splits"]),
          "a frozen repeated split no longer favors correction")
    check({row["n_cells"] for row in results["dense_equivalence"]} == {64, 128, 256},
          "dense-control sizes drift")
    check(max(row["relative_solution_difference"] for row in results["dense_equivalence"]) < 1e-7,
          "dense-equivalence gate failed")
    check({row["right_hand_sides"] for row in results["solver_runtime"]} == {1, 2, 4, 8, 12},
          "solver block-size audit drift")
    check(max(row["relative_solution_difference"] for row in results["solver_runtime"]) < 1e-12,
          "scalar/synchronized equivalence gate failed")
    q12 = next(row for row in results["solver_runtime"] if row["right_hand_sides"] == 12)
    check(q12["batched_operator_launches"] < q12["scalar_operator_launches"],
          "synchronized launch-count claim failed")
    check(all(value is False for value in results["claim_boundaries"].values()),
          "a prohibited broad claim is enabled")


def _scientific_projection(results: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in results.items()
        if key not in {"environment", "solver_runtime"}
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_replay(results: dict[str, object]) -> None:
    replay_results_path = REPO / "code/results/replay_env_b.json"
    replay_manifest_path = REPO / "code/results/replay_manifest.json"
    replay_results = json.loads(replay_results_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    comparison = replay["direct_a2_b_comparison"]
    artifacts = replay["frozen_artifacts"]
    reference = replay["reference_environment"]
    left = _scientific_projection(results)
    right = _scientific_projection(replay_results)

    check(replay["schema_version"] == 1 and replay["status"] == "PASS",
          "replay manifest status drift")
    check("not an independent scientific replication" in replay["interpretation"],
          "replay interpretation boundary missing")
    check(reference["version_locked"] is True and reference["hash_locked"] is False,
          "replay lock disclosure drift")
    check(reference["runtime_lock_sha256"] == sha256(REPO / reference["runtime_lock"]),
          "replay runtime-lock digest drift")
    check(artifacts["experiment_source_sha256"]
          == sha256(REPO / "code/src/mfcell/experiment.py"),
          "replay source digest drift")
    check(artifacts["locked_results_sha256"]
          == sha256(REPO / "code/results/results.json"),
          "replay locked-result digest drift")
    check(artifacts["environment_b_results_sha256"] == sha256(replay_results_path),
          "environment-B result digest drift")
    check(re.fullmatch(r"[0-9a-f]{64}", artifacts["final_wheel_sha256"]) is not None,
          "final-wheel digest is malformed")
    check(comparison["excluded_top_level_fields"] == ["environment", "solver_runtime"],
          "replay projection exclusion drift")
    check(comparison["per_gene_metrics_included"] is True
          and comparison["per_gene_differences_exact"] is True
          and comparison["method_spearman_included"] is True
          and comparison["reported_metrics_excluded"] is False,
          "replay metric-coverage disclosure drift")
    check(left == right and comparison["exact_projection_equal"] is True,
          "A2/B scientific projections are not exactly equal")
    left_digest = _canonical_sha256(left)
    right_digest = _canonical_sha256(right)
    check(left_digest == comparison["environment_a2_projection_sha256"],
          "A2 projection digest drift")
    check(right_digest == comparison["environment_b_projection_sha256"],
          "B projection digest drift")


def validate_hygiene() -> None:
    for path in REPO.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir():
            check(path.name not in FORBIDDEN_DIRS, f"forbidden directory: {path}")
            check(not path.name.endswith(".egg-info"), f"egg-info directory: {path}")
            continue
        check(path.name not in {".DS_Store", "mainNotes.bib"}, f"forbidden file: {path}")
        check(not any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES),
              f"build artifact: {path}")
        if path.stat().st_size <= 2_000_000:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            check(not re.search(r"/Users/[A-Za-z0-9._-]+/", text),
                  f"absolute private path: {path}")
            private_key_marker = "-----BEGIN " + "PRIVATE KEY-----"
            check(private_key_marker not in text, f"private key marker: {path}")
            check(not re.search(r"\bAKIA[0-9A-Z]{16}\b", text), f"AWS key marker: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = json.loads((REPO / "code/results/results.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    validate_data(results, manifest)
    quantities = validate_manuscript(manifest)
    validate_results(results, manifest)
    validate_replay(results)
    validate_hygiene()
    print(json.dumps({
        "status": "PASS",
        **quantities,
        "manifest_artifacts": len(manifest["artifacts"]),
        "primary_extra_trees_mse": results["primary"]["metrics"]["extra_trees"]["mse"],
        "primary_method_mse": results["primary"]["metrics"]["method"]["mse"],
        "external_extra_trees_mse": results["external"]["metrics"]["extra_trees"]["mse"],
        "external_method_mse": results["external"]["metrics"]["method"]["mse"],
    }, indent=2))


if __name__ == "__main__":
    main()
