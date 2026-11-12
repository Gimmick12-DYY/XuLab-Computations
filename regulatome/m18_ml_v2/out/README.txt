m18_ml_v2: Regulome model version 2 with richer inputs. Cross-validated, benchmark-developed (the m7 test chromosomes were scored once before; nothing here is an untouched test). HEK293T.
SPEC.md + Amendments 1-3 (commits 4d82b38, b7989fa, 6e46a13, 093037d). Folds: folds.tsv, sha256 in folds.tsv.sha256 (a0e8281, before any fit). Fit script m18_fit.py committed cb4c7f9 before any result.
Train/test separation (code tags in m18_fit.py):
- Whole-chromosome 5-fold CV; neighbouring and shared-promoter genes stay on one chromosome, so in one fold.
- LEAK-GUARD-1 centre(): per-TF target mean from training rows only. LEAK-GUARD-2: elastic-net scaler fitted on training rows only. LEAK-GUARD-3: grid and early stopping on an inner validation fold inside the training folds, then refit on the training folds.
- Percentile features (block B) are genome-wide and target-free (m18_features.py).
- Leak test: target permuted within TF inside the training folds -> out-of-fold Spearman (ladder.tsv rung full_permtarget).
- Block R (NTC expression) computed from a random half of the NTC cells (seed 20261112); the other half reported as *_half1. The strict version (z recomputed against one NTC half) was not run (needs the full m4 kd_response rerun).
- By-TF split: leave one of the 10 primary TFs out AND its chromosome fold out; block E and the TF own expression kept (keepE) or dropped (dropE).
Blocks: A gene baseline (+E TF descriptors); R NTC pseudo-bulk expression; H histone = H3K27ac only (already in A), so A+R+H = A+R; C own motif (no HITAG); B/BC this TF binding and motif x binding;
D binding of all 79 TFs at the gene; F coordinated binding (Figure 3 modules, hubs, Cicero edges, super-enhancers); G complex pairs (provisional candidate list, sha256 08fe2064...).
Outputs (out/): ladder.tsv (per rung mean per-TF out-of-fold Spearman, TF-bootstrap 95% interval, difference from the previous rung), contrasts.tsv (declared readings),
per_tf_oof_primary.tsv / per_tf_oof_all.tsv (per-TF Spearman per config beside reliability and sqrt(reliability)), references.tsv (unfitted references), runs/ (per config: per-TF table, folds, out-of-fold predictions,
and for primary LightGBM full: per-gene feature contributions *_contrib.npz and per-TF mean |SHAP| by block).
