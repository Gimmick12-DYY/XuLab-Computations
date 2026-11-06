# R2b specific-residual target lists (manifest v4)
Script r2b_residual.py, run by r2b_residual.sbatch; spec and implementation notes in SPEC.md.
Outputs: per_tf/<TF>.tsv, r2b_summary.tsv. Freeze: manifest_v4.sha256 (sha256 of all outputs, the script and SPEC.md), written by the same job; manifests v1-v3 are untouched.
Decisions confirmed in the design thread on 2026-11-06 before the run: expected reads = the Gate 1 v2.1 Poisson GLM, unchanged; split halves = the first Gate 1 v2.1 split (seed 20261029 + TF index), per-half expectation scaled by that half's reads.
No Perturb-seq data is read here; the Perturb-seq comparison is a later, separate step.
