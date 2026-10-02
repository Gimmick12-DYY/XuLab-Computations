# Novel Motif Finding for Unknown TFs — roadmap

Enhancement layer on the existing de-novo discovery (`slurm/motif_denovo_pmat.sbatch`:
HOMER + STREME + TOMTOM for the ~45 TFs with no HOCOMOCO/HOMER motif). Goal: turn
raw "unknown"/novel de-novo motifs into **validated, TF-assigned** motifs.

Status legend: ✅ built · 🟡 partial/needs data · ⬜ not started.

## Step 1 — Re-annotate HOMER "unknown" motifs  🟡
- Switch HOCOMOCO v11 → **v12 + Codebook** (CisBP 3.1 / Zenodo). Build the merged DB
  that TOMTOM already consumes (`cache/motifdb/merged_human_motifs.meme`):
  - ✅ `scripts/merge_meme_db.py` — merge several MEME DBs, tag by source.
    ```
    python downstream/scripts/merge_meme_db.py \
      --in HOCOMOCOv12_H12CORE_meme_format.meme=h12 codebook_cisbp3.1.meme=cdbk \
      --out downstream/cache/motifdb/merged_human_motifs.meme
    ```
  - ⬜ download helpers: HOCOMOCO v12 (hocomoco12.autosome.org, MEME export) +
    Codebook motifs (Zenodo/CisBP 3.1). **Need URLs confirmed.**
- Compare by **affinity correlation (MoSBAT)** instead of HOMER/TOMTOM default. Current
  de-novo uses `tomtom -dist pearson` (reasonable proxy). ⬜ MoSBAT is the upgrade.
- Expectation: many "novel" motifs map to Codebook TFs (C2H2-ZNF, CXXC, AT-hook, BED-zf).

## Step 2 — Filter artifacts  🟡
- ✅ Contaminant filter: `scripts/flag_motif_artifacts.py` flags de-novo motifs whose best
  TOMTOM match (vs v12+Codebook) is CTCF/NFY/YY1/SP-KLF/ETS (`--families` accepts the
  Codebook set verbatim). Joins onto the Step-3 table → `pass` column; shortlist =
  **high AUROC AND pass**. Wired into `rank_motifs.sbatch` (step 4 of the script).
- 🟡 **GC + width-matched shuffled background**: emit_bins_bed.py gives OCR-matched bg;
  HOMER `-useNewBg`. ⬜ add explicit GC+width-matched shuffle for the scoring/enrichment.
- 🟡 Repeat/TE clustering: logic in `flag_motif_artifacts.py` (`--repeat-bed` +
  `--instances-bed` → `te_clustered`); ⬜ needs a RepeatMasker BED (not yet available).

## Step 3 — Rank by PERFORMANCE, not information content  ✅
- ✅ `scripts/rank_motifs_by_performance.py` — per motif, best PWM log-odds per sequence
  (both strands), then **AUROC + AUPRC** of pos vs neg bins; reports info content too, so
  a low-IC "flat" motif with high AUROC is visible. (Validated on synthetic: real motif
  AUROC 1.0, flat 0.5.)
- ✅ `slurm/rank_motifs.sbatch` — scoring set = the TF's binding BINS (out-of-representation
  vs the pmat PEAKS used for discovery → held-out-ish); candidates = HOMER de-novo + STREME.
- Rank by AUROC/AUPRC, **not** HOMER p-value or logo sharpness.

## Step 4 — Cross-dataset validation (triple overlap)  ⬜
- Intersect: our imputed peaks ∩ Codebook ChIP-seq / GHT-SELEX peaks ∩ PWM matches (FIMO).
- Set each threshold to maximize Jaccard; keep only motifs reproducible across data types.
- **Needs:** Codebook ChIP-seq / GHT-SELEX peak sets (download).

## Step 5 — Assign orphan motifs to candidate TFs  ⬜
- Correlate motif activity (**chromVAR** or MARA) with candidate TF expression across
  cells/clusters; restrict candidates to the **175 TFs still lacking motifs**.
- Check motif shape fits the TF's DNA-binding-domain class; test **phyloP** conservation at
  motif sites. Optional: AlphaFold3 structure, allele-specific accessibility.
- **Needs:** chromVAR, per-cell/cluster TF expression (RNA), phyloP bigwig, DBD-class table.

## Build order
Step 3 (✅ done, self-contained, methodological centerpiece) → Step 1 DB (confirm URLs) →
Step 2 filters → Step 4 (needs Codebook peaks) → Step 5 (needs chromVAR + expression).
