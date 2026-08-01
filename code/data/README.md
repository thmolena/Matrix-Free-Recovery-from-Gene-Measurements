# Data provenance

The package contains the unchanged filtered gene-barcode archives for two 10x
Genomics public datasets:

| dataset | upstream landing page | archive bytes | matrix shape | nonzeros | recorded molecules |
|---|---|---:|---:|---:|---:|
| PBMC3k | https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0 | 7,621,991 | 32,738 x 2,700 | 2,286,884 | 6,390,631 |
| PBMC4k | https://www.10xgenomics.com/datasets/4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0 | 18,423,814 | 33,694 x 4,340 | 5,727,695 | 18,874,824 |

Direct archive URLs, full SHA-256 digests, matrix dimensions, nonzero counts,
and molecule totals are hard-coded in the loader. A mismatch aborts replay.
The same archives are mirrored in `src/mfcell/data/source/` so an ordinary
wheel can verify offline. Run `mfcell-reproduce --verify` to authenticate and
recompute the complete experiment.

Both upstream dataset pages identify the archives as CC BY 4.0. See
`THIRD_PARTY.md`; the repository MIT license applies only to original software.
