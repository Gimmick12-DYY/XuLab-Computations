# R2b specific-residual target lists, v5 (manifest v5)
The 1/201 floor was a numerical resolution limit; rule unchanged; decided before any comparison; v4 kept as record.
v5 differs from v4 (regulatome/r2b_residual/, manifest v4, unchanged) only in the number of Poisson null draws: 10,000 instead of 200 (design-thread ruling, 2026-11-06). Same list rule (S_exp >= 5, q <= 0.10, both halves > 0, top 1,000 by X), same Gate 1 v2.1 GLM expectation, same first split, same fixed seeds (null draws are taken in the same order, so the first 200 equal v4's).
Script: r2b_residual.py here (n_draws, output folder and manifest label as arguments), run by r2b_residual.sbatch. Spec: ../r2b_residual/SPEC.md. Outputs: per_tf/<TF>.tsv, r2b_summary.tsv; freeze: manifest_v5.sha256 (outputs, script, SPEC.md, this README), written by the same job.
No Perturb-seq data is read here.
