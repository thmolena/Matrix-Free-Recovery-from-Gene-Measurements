# Leakage-Audited Matrix-Free Residual Propagation for Gene Measurements

**Project page:** [https://thmolena.github.io/Matrix-Free-Recovery-from-Gene-Measurements/](https://thmolena.github.io/Matrix-Free-Recovery-from-Gene-Measurements/)

This repository contains a one-column research manuscript and an
installable CPU reproduction package for a leakage-audited, two-dataset
single-cell expression-prediction study.

- Manuscript: [`main.pdf`](main.pdf)
- Source: [`main.tex`](main.tex)
- Code: [`code`](code)
- Authentic data: unchanged 10x Genomics PBMC3k and PBMC4k filtered
  gene-barcode archives, with full SHA-256 digests in
  [`code/data/checksums.sha256`](code/data/checksums.sha256).

## Result

The repaired protocol predicts 12 prespecified marker genes from 96 disjoint
input genes. PBMC3k supplies model and graph-weight selection; all selected
settings are frozen before the PBMC4k transfer test. Normalization excludes
all target counts from the library-size denominator, and residual seeds for
the flexible base learner are cross-fitted.

| test | ExtraTrees MSE | corrected MSE | relative change |
|---|---:|---:|---:|
| PBMC3k | 0.416046 | **0.398645** | -4.18% |
| PBMC4k frozen transfer | 0.418608 | **0.412775** | -1.39% |

The previous graph method lost to a tuned ExtraTrees baseline and its
superiority claim is withdrawn. The repaired correction wins on all five
additional PBMC3k split audits, but those splits reuse cells and are not
independent biological replicates.

The recovery path applies the normal matrix through sparse products and never
forms a dense cell-by-cell operator. All final PCG solves converge with maximum
relative residual below `1e-9`. Synchronized multi-output PCG reduces sparse
operator launches; it is faster here for 8 and 12 outputs but slower for 1--4,
so no universal or end-to-end speedup is claimed.

## Install and verify

The validated reference is CPython 3.13.12 on macOS 26.6 arm64 using the
version-locked PyPI numerical wheels in `code/requirements-lock.txt` and NumPy's Accelerate
BLAS/LAPACK backend. Conda or another BLAS build can differ even when version
strings match. Discrete protocol fields and digests are checked exactly;
floating evidence uses a disclosed 0.5% relative and `5e-5` absolute envelope
plus explicit scientific gates. Only environment and runtime are excluded from
the locked projection.
Tree fitting and the OpenMP, OpenBLAS, MKL, and NumExpr pools use one thread in
the registered replay to prevent parallel scheduling or reduction drift.
The lock itself has SHA-256
`b1b02d84e0991707d67530775a1c01750ce982c8c6a12f30757e23e8a6ad9db9`;
it has exact versions but, because dependency-wheel downloads were unavailable
during release assembly, it is explicitly not a hash-locked requirements file.

```bash
python -m pip install --only-binary=:all: -r code/requirements-lock.txt
python -m pip install --no-deps ./code
mfcell-reproduce --verify
```

Editable installation is also supported:

```bash
python -m pip install --only-binary=:all: -r code/requirements-test-lock.txt
python -m pip install --no-deps -e ./code
python -m pytest code/tests
```

The ordinary wheel includes both source archives and locked scientific results,
so verification works offline after installation.
Two independently created clean software environments produced exactly equal
scientific projections, including every per-gene metric. The preserved B output
and comparison record are `code/results/replay_env_b.json` and
`code/results/replay_manifest.json`; this is software reproduction, not an
independent scientific replication.

## Scope of evidence

PBMC3k and PBMC4k are authentic public cross-sectional datasets, each from one
donor. The paper establishes finite-dimensional properties of the declared
quadratic solve and reports dataset-specific prediction results. It does not
claim worldwide novelty for residual propagation, donor generalization,
biological recovery, causal transitions, universal baseline superiority,
quantum advantage, or an end-to-end speedup. Numerical experiments check the
implementation; they are not mathematical proofs.

## License

Code is released under the repository MIT license. The bundled 10x archives
are unchanged and remain under CC BY 4.0 with upstream attribution.
