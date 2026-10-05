# Novel Motif Finding for Unknown TFs — roadmap

Enhancement layer on the existing de-novo discovery (`slurm/motif_denovo_pmat.sbatch`:
HOMER + STREME + TOMTOM for the ~45 TFs with no HOCOMOCO/HOMER motif). Goal: turn
raw "unknown"/novel de-novo motifs into **validated, TF-assigned** motifs.

Status legend: ✅ built · 🟡 partial/needs data · ⬜ not started.

## Data sources (confirmed)
- **HOCOMOCO v12** — hocomoco12.autosome.org (MEME export). *You have this.*
- **Codebook / MEX** (Hughes et al., *Nature* 2026, "An expanded codebook…") —
  Zenodo **10.5281/zenodo.15667805**, browser mex.autosome.org. ✅ `scripts/fetch_codebook.sh`
  pulls: `MEX_artifacts_formatted.tgz` (37 artifact motifs, MEME), `MEX_top1.zip` (1 curated
  motif/TF, 204 TFs, `.ppm`), `metadata_complete_motif.zip`. ChIP-seq peaks = `MEX.CHS.tar`
  (`CHS=1`); GHT-SELEX = zenodo 8327970; raw SRA PRJEB78913/76622/61115.

## Step 1 — Re-annotate HOMER "unknown" motifs  ✅ (MoSBAT ⬜)
- ✅ `scripts/fetch_codebook.sh` → `codebook_top1.meme`; ✅ `scripts/ppm_to_meme.py` converts
  Codebook `.ppm/.pcm` → MEME (TF = leading token). ✅ `scripts/merge_meme_db.py` builds the
  v12+Codebook DB that TOMTOM consumes:
  ```
  bash motif_analysis/scripts/fetch_codebook.sh
  python motif_analysis/scripts/merge_meme_db.py \
    --in HOCOMOCOv12_H12CORE_meme_format.meme=h12 \
         motif_analysis/cache/motifdb/codebook_top1.meme=cdbk \
    --out motif_analysis/cache/motifdb/merged_human_motifs.meme
  ```
- ⬜ MoSBAT (affinity correlation) as the upgrade over `tomtom -dist pearson`.
- Expectation: many "novel" motifs map to Codebook TFs (C2H2-ZNF, CXXC, AT-hook, BED-zf).

## Step 2 — Filter artifacts  ✅ (GC-bg ⬜, TE needs instances)
- ✅ **Codebook artifact set** (their empirical one): `fetch_codebook.sh` →
  `codebook_artifacts.meme`; `flag_motif_artifacts.py --artifact-tomtom` flags any de-novo
  motif matching it (poly-G, Alu/repeat, CAC/GGAA, NFI, self-annealing, …). Plus the
  name-regex contaminant fallback (CTCF/NFY/YY1/SP-KLF/ETS, `--families` overrides). Joins the
  Step-3 table → `pass` (= not contaminant, not Codebook-artifact, not TE-clustered).
  Shortlist = **high AUROC AND pass**. Wired into `rank_motifs.sbatch`.
- ✅ RepeatMasker: `scripts/fetch_hg38_rmsk_bed.sh` (UCSC hg38 rmsk → BED); auto-used when
  present. 🟡 TE flag also needs `INSTANCES_BED` (FIMO motif hits) — not yet generated.
- 🟡 **GC + width-matched shuffled background**: emit_bins_bed gives OCR-matched bg; ⬜ add
  explicit GC+width-matched shuffle.

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
