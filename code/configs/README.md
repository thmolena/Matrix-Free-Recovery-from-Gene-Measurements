# Configurations

The audited experiment is intentionally code-registered in
`src/mfcell/experiment.py`: dataset digests, split seeds, target Ensembl IDs,
baseline grids, graph weights, and solver tolerances are visible constants.
PBMC3k development data select the configuration; test outcomes are never
configuration inputs. The complete selected and rejected rows are recorded in
`results/results.json`.
