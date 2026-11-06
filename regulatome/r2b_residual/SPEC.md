# R2b specific-residual target lists (approved by Yuyang in the design thread, "Yep let's try it", 2026-11-06 19:57Z). Spec, verbatim:

INPUTS (read only): Gate 1 v2.1 per-element observed reads per TF and expected reads (pooled-panel accessibility scaled to TF depth), plus its split halves A/B of each TF's cells — reuse Gate 1 v2.1 code. ABC links regulatome/m2_abc/ (104,071 links, same table R2 used, kappa 0). Gene universe = genes with >=1 ABC link.
STAT per TF q, gene g: S_obs = sum_e obs(q,e)*ABC(e,g); S_exp = sum_e exp(q,e)*ABC(e,g) (raw counts). X = log2((S_obs+0.5)/(S_exp+0.5)); score only S_exp>=5. Null: 200 draws obs(q,e)~Poisson(exp(q,e)), recompute X; p = (1+#X_null>=X)/201; BH over scored genes per TF. Halves: X_A, X_B with S_exp>=2.5 per half; require both >0.
LIST: q<=0.10 and both halves >0, ranked by X, cap 1,000; keep uncapped count and rank; no ATAC filter; zero-length lists valid.
OUTPUT: regulatome/r2b_residual/per_tf/<TF>.tsv (gene,S_obs,S_exp,X,p,q,X_A,X_B,pass) and r2b_residual/r2b_summary.tsv (TF, tier, n_scored, n_pass, n_list, median X of list, overlap with direct top 1,000, overlap with ATAC-only top 1,000).
FREEZE: manifest v4 = sha256 of all r2b_residual outputs + script, written in the same job before anything else; v1-v3 untouched.
RESOURCES: one sbatch, 1 CPU, 16 GB, <45 min, new folder regulatome/r2b_residual/ only. Vectorise per TF.
Do NOT read Perturb-seq anywhere. The second Perturb-seq comparison is a later, separate step.

## Implementation notes (where the code had to choose)
- Expected reads: Gate 1 v2.1 computes them with its Poisson GLM (offset = TF reads; covariates ATAC CPM, leave-TF-out pooled panel CPM, width, GC, GC^2, mappability), not with pooled-panel CPM alone. The script reuses that GLM unchanged (design() and glm_mu() copied from gate1/v2_1/gate1_v21.py).
- Halves: Gate 1 v2.1 does 20 random splits per TF and saves no split. The script uses its first split (same seed rule, seed 20261029 + TF index, first permutation), with expected per half = mu x half reads / TF reads, as in v2.1.
- Elements and links: active elements (promoter, distal_active, bivalent) and power-law ABC links at 0.02 mapped to them, exactly as Gate 1 v2.1 and R2 do; the number of mapped links is printed.
- Direct top 1,000 = regulatome/r2_targets/top1000/<TF>.tsv (R2's frozen list). ATAC-only top 1,000 = top 1,000 genes by S_atac in regulatome/r2_targets/per_tf/<TF>.tsv.gz.
