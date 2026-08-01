# `mfcell` reproduction package

This CPU package reproduces the manuscript experiment on the original 10x
Genomics PBMC3k and PBMC4k filtered gene-barcode matrices. It predicts 12
prespecified marker genes from 96 disjoint input genes, cross-fits ExtraTrees
residual seeds, and applies the normal operator through sparse matrix-vector or
matrix-block products solved by Jacobi-preconditioned conjugate gradients.

## Install and verify

From the repository root, the canonical installation is:

```bash
python -m pip install --only-binary=:all: -r code/requirements-lock.txt
python -m pip install --no-deps ./code
mfcell-reproduce --verify
```

Locked verification targets CPython 3.13.12 on macOS 26.6 arm64, the
version-locked PyPI wheels in the lock file, and NumPy's Accelerate BLAS/LAPACK backend. Other binary
backends may differ even when version strings match. Discrete protocol fields
and digests match exactly; floating evidence has a 0.5% relative and `5e-5`
absolute envelope plus explicit gates. Only environment and runtime are
excluded from the locked projection.
Tree fitting and the OpenMP, OpenBLAS, MKL, and NumExpr pools use one thread in
the registered replay to prevent parallel scheduling or reduction drift.
The lock SHA-256 is
`b1b02d84e0991707d67530775a1c01750ce982c8c6a12f30757e23e8a6ad9db9`.
It has exact versions but is not hash-locked; this limitation is explicit.

For editable development and artifact regeneration:

```bash
python -m pip install --only-binary=:all: -r code/requirements-test-lock.txt
python -m pip install --no-deps -e ./code
mfcell-reproduce --write
python -m pytest code/tests
```

The ordinary wheel contains both unchanged source archives and the locked
scientific results, so `mfcell-reproduce --verify` works offline after
installation.

## Reproducibility map

- `src/mfcell/`: matrix-free solver and experiment.
- `tests/test_reproduction.py`: source, operator,
  certificate, and locked-result tests.
- `data/source/`: authenticated PBMC3k and PBMC4k archives.
- `results/`: locked JSON/TeX outputs plus the preserved clean-environment-B
  replay and direct A2--B comparison manifest. The equal software replay is not
  an independent scientific replication.
- `manuscript_assets/figures/`: five generated scientific figures.
- `manuscript_assets/tables/`: numerical tables generated from the locked run.
- `manuscript_assets/evidence_manifest.json`: data and artifact provenance.
- `manuscript_assets/validate_release.py`: public-package contract validator.

## Claim boundaries

Each archive represents one cross-sectional donor. PBMC4k is a frozen-parameter
transfer test, not evidence of population-level donor generalization. The
experiment does not establish lineage, biological recovery, causality,
worldwide algorithmic priority, quantum advantage, or a universal end-to-end
speedup. Solver timings are bounded diagnostics and excluded from locked
scientific-result verification.

The MIT license applies to the code. The bundled 10x archive remains subject
to its source terms and attribution.

<!-- standardized-public-entry-points -->
## Standard public entry points

From the repository root:

```bash
python -m pip install --only-binary=:all: -r code/requirements-lock.txt
python -m pip install --no-deps -e code
python code/scripts/download_data.py  # authenticate bundled archives
python code/scripts/reproduce.py
python code/scripts/make_figures.py
python code/scripts/validate_release.py
```

The wrappers delegate to the package's existing scientific entry points.
Dataset provenance is in `data/`, locked evidence in `results/`, and
manuscript bibliography/figures in `manuscript_assets/`.
