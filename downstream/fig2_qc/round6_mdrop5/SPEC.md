# Round 6b rerun with the Mdrop5-masked imputed data (declared 2026-11-20, before any result)

User request 2026-11-20: rerun the differential-peak 6x6 vs bulk ChIP (round 6b) with the newly masked imputed data.
- **Mask.** Newest imputed mask on Longleaf = m46 Amendment 5 (regulatome/m46_stage1_imputed/work/masks_extra.npz,
  written 2026-11-18): Mdrop5 = 1 kb bins with mean ATAC CPM > 0.0781542 over the two pooled ATAC bigwigs (the September
  definition, downstream/apply_atac_cpm_drop5_mask.py). No mask file is newer than 2026-11-18. (The per-TF MdropTF mask
  from the same amendment is derived from our lab peaks and is not used here.)
- **Imputed inputs (r20a_imp_mdrop5.py).** Per TF, from the genome-wide pre-mask matrix (impute/matrix_csr.npz.pre_openmask)
  and the raw cell x bin matrix (mm): model-added calls = pre-mask calls minus raw-positive cells.
  - masked = raw-positive cells + model-added calls inside Mdrop5 (raw always kept, as in m46) -> inputs/imp79_mdrop5.npz
  - added  = model-added calls inside Mdrop5 only (no copied raw reads) -> inputs/imp79_mdrop5_added.npz
  ZNF768 has no pre-mask file; its current ATAC-union-masked matrix is used (flagged; it only enters the 78-TF pool).
- **Tests.** Identical to round 6b (r14_imputed_diff.py, unchanged except the input file and output folder):
  I1 = differential regions called from the imputed counts (binomial vs the other 78 TFs on the round 6 candidate rows,
  q < 0.05, log2FC > 1) scored with imputed counts (M1 overlap enrichment, M2a per-region Pearson, M2b 30 summit
  groups); I2 = raw-called D-B sets scored with imputed (M2a, M2b); I3 = I1 sets scored with raw (M2a, M2b). Top 1,000
  and 5,000; CTCF references ENCODE HEK293 and GSE103651 HEK293T nexus; halves all / odd / even + X.
- **Statistics.** Own-best rows/columns, diagonal minus off-diagonal, exact 720-permutation p, without CTCF (120), halves.
- **Extra.** Fraction of differential-region midpoints inside the HEK293T ATAC union (r20c_atac_frac.py), for D-B, the
  6b D-I sets, and both Mdrop5 arms.
- Outputs results_293T_analysis/fig2_qc/round6_diffpeaks/imputed_mdrop5/{masked,added}/; Mac
  figures_fig2_qc/round6_diffpeaks/imputed_mdrop5/.
