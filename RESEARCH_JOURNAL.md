# Research Journal — XuLab-Computations

A durable, chronological record of **major** updates: what changed, why, and the
outcome/decision. This is the place to review the arc of the project later.

**Scope.** Single-cell TF binding for the ~78-TF HEK293T scCUT&Tag "TF1000cells"
panel: imputation of sparse per-TF binding, benchmarking against bulk, and
downstream biology (peak calling, motif enrichment, TF co-occupancy / complexes,
A/B compartments, ChromHMM state analysis).

**How to use this file.**
- Newest entries at the top of the log (reverse-chronological).
- One entry per *major* update (a new pipeline, a resolved confound, a corrected
  method, a key biological finding). Routine commits stay in git.
- Each entry: date, one-line title, **Context / Change / Outcome** (+ **Files** or
  **Commit** when useful). Convert relative dates to absolute.
- Seeded 2026-09-13 from git history, the memory notes, and
  `CHANGES_July_to_August_2026.md` (which remains the detailed July→mid-Aug record).

**Environment note.** Code runs on the Longleaf cluster
(`/work/users/d/y/dyy12/XuLab`) via commit → push → pull; a stale checkout on the
cluster is the usual cause of "unchanged output."

---

## Log (newest first)

### 2026-09-23 — SpQN result: complexity confound removed + new candidate complexes (for motif confirmation)
- **Status:** SpQN on the A/B-compartment TF×TF correlation **removed the cell-count
  confound.** On the delivered genome-scope matrix, `corr(cells, mean_corr) = +0.00`
  (linear) / `+0.09` (vs log10 cells) ≈ 0. Concrete proof: RBBP4 (181,396 cells)
  mean-corr **0.44** vs HOXC4 (1,023 cells) **0.58** — the deepest TF is *less* sticky
  than a shallow one, so the "deep-TF rich club" is gone. (Still want the pre-SpQN
  `+X` from the `05_compartment_rpkm` `.out` `confound` line to headline the delta.)
- **Data analyzed:** `compartment_counts_genome.tsv` = the SpQN-corrected TF×TF matrix,
  **percentile-transformed** (off-diag quantiles uniform `[.05,.25,.50,.75,.95]`), 79 TFs.
  Values below are mean pairwise **percentile** of Pearson r (background median = 0.50;
  ≥0.90 = strong co-occupancy). Method: Bron–Kerbosch cliques + reciprocal-best-partner
  pairs + per-module mean-percentile scoring.

- **Validation (positive controls):**
  - PRC2 `EZH2+MTF2` = **0.98** ✓
  - Cohesin/insulator `CTCF+STAG2` = **0.93** ✓
  - NuRD core `RBBP4+RBBP7+MBD3` = **0.60** ✗ — NuRD does NOT co-occupy at compartment
    scale (expected; NuRD is peak-scale → check via the Cicero/cobinding track).

- **NEW candidate complexes/modules (ranked; to confirm by motif co-enrichment):**
  1. **G9a/GLP H3K9-methylation** — `WIZ + ZNF644` = **0.98** (mutual best partners).
     Both are documented G9a–GLP (EHMT2/EHMT1) components → strongest, literature-backed
     de-novo hit.
  2. **Repressive-chromatin super-module** — `WIZ + ZNF644 + MTF2 + SMARCA1` = **0.99**
     (H3K9 + H3K27/PRC2 + ISWI remodeling co-occupying the same territory).
  3. **CTCF-independent boundary** — `MAZ + ZNF143` = **0.98**, while `MAZ+ZNF143+CTCF`
     = **0.40** (CTCF does NOT join → matches MAZ-as-CTCF-independent-boundary reports).
  4. **Metabolic** — `FOXK1 + MLX` = **0.99** (extends to SREBF2 = 0.89). MLX↔MondoA/ChREBP,
     FOXK1 glycolysis regulator.
  5. **Basal/NC2 repressor** — `DR1 + POGK + ZNF746` = **0.98** (DR1 = NC2β).
  6. **NuRD/DREAM-adjacent pairs** — `RBBP7 + ZNF703` = 0.98 (ZNF703 recruits NuRD);
     `E2F3 + RBBP4` = 0.998 (RbAp48 in Rb–E2F/DREAM).
  - Other strong reciprocal pairs to sanity-check: `SRCAP–MLX` 0.998, `NFYA–AKAP8L` 0.984,
    `MBD3–ZNF777` 0.962, `NME2–NFYA` 0.959, `SOX4–TRAFD1` 0.996.

- **Caveats (state in any writeup):** compartment-scale co-occupancy = shared large-scale
  territory, **not** proof of physical interaction; some high pairs are shared-territory
  coincidence (e.g. `RBBP4–XPA` 1.00). The **A/B-activity confound** still shapes the big
  blocks — trust the *within-block tight pairs above*, not the blocks. SpQN can over-flatten
  at n=78; validation complexes surviving is the reassurance.

- **NEXT (cluster agent):** confirm the ranked modules by **motif co-enrichment** — do the
  paired TFs' motifs co-occur in the same regions above background? Prioritize the two clean
  literature-backed hits **WIZ+ZNF644 (G9a/GLP)** and **MAZ+ZNF143 (CTCF-independent
  boundary)**. Cross-check all strong pairs against CORUM/BioGRID (annotated vs novel), and
  against the peak-level Cicero/cobinding results (esp. NuRD, which should appear there but
  not here). Motif pipeline: `downstream/slurm/motif_enrichment.sbatch` (AME `--scoring avg`).
- **Files/inputs:** `compartment_counts_genome.tsv`, `data/TF1000cells.meta.csv`,
  `tf_complex/scripts/compartment_rpkm_correlation.py` (`--spqn`).

### 2026-09-23 — Compartment correlation: cell-count normalization is a no-op; wired SpQN (with caveat)
- **Context:** The A/B-compartment TF×TF correlation has a complexity bias — TFs
  with more cells correlate with each other. PI suggested normalizing by cell count.
- **Finding:** Dividing each TF by its cell count is a **per-TF scalar → Pearson
  scale-invariant → no-op** (the new `per_cell` transform gives a byte-identical
  matrix to RPKM; same as CPM). The bias is not magnitude but **differential
  measurement noise**: low-cell TFs have noisy profiles → attenuated correlations;
  high-cell TFs are clean → correlate strongly with each other. Scaling can't touch
  signal-to-noise. Real fixes: equalize cells by **downsampling**, or **SpQN**.
- **Change:** Wired **SpQN** (Wang/Hicks/Hansen 2022) into
  `compartment_rpkm_correlation.py` (`--spqn`, reuses the vetted `tf_complexes.spqn`,
  covariate = per-TF cell count) + `_confound` diagnostic (`corr(cells, mean_corr)`
  before/after) + `SPQN=1` sbatch knob. Writes `spqn_similarity_<scope>.tsv`;
  clusters/plots the corrected matrix.
- **Caveat (validated on synthetic):** at ~78 TFs SpQN is coarse and its
  level-to-reference mechanism raised the off-diag median (+0.65→+0.85) while only
  partly cutting the confound (+0.66→+0.48) and **flattening a planted complex**
  (gap 0.18→0.04). So it may make the matrix uniformly red and wash out NuRD/PRC2
  rather than surface them. **Judge on real data via the diagnostic + whether
  NuRD (rbbp4/rbbp7/mbd3) and PRC2 (ezh2/mtf2) stand above background;** if not,
  fall back to downsample-to-equal-cells. Separate confound (all TFs prefer active
  compartments) still needs a per-domain enrichment ratio.
- **Files:** `tf_complex/scripts/compartment_rpkm_correlation.py`,
  `tf_complex/slurm/05_compartment_rpkm.sbatch`.

### 2026-09-23 — Cicero co-binding: run on the fragment-called peak matrix (the real fix)
- **Context:** After adding significance (below), a validation run still barely
  matched the lab gold — pair Jaccard 0.028, ~7k regions/5k edges vs gold ~20k/17k.
  Significance was **orthogonal**; the real problem was the **input peak universe**.
- **Diagnosis chain:**
  - Our cobinding Cicero was fed peaks from `call_raw_peaks.py`, which calls MACS on
    the **1 kb-bin** pseudobulk (synthetic fragments at bin midpoints). That quantizes
    peaks to a 1 kb grid → **0 exact coordinate matches** with the lab's fragment
    peaks, and ~⅓ as many sites.
  - The lab's gold peaks are variable-width fragment MACS peaks (min 150, median 262).
    Resolution itself is *not* the blocker: gold peaks collapse onto 1 kb bins nearly
    losslessly (19,199 bins, 0.6% pairs lost) — so a bin-level analysis *could*
    recover ~99%; the gap was sensitivity/wrong sites, not the 1 kb grid.
  - We do **not** have raw fragments. But the colleague sent a **fragment-called peak
    matrix**, `data/TF1000cells.pmat.mtx.rds` (237,450 peaks × 885,203 cells, all TFs
    pooled, mean width 412 bp) — the correct universe we'd been ignoring.
- **Change:** New `cobinding/scripts/export_pmat_tf_mm.R` subsets the pmat to a TF's
  cells (barcodes from `unified/work/<tf>/mm/barcodes.tsv.gz`), filters
  `min-cells-per-peak`, and writes the `mm/{matrix,regions,barcodes}` trio that
  `02_build_cds.R` consumes. `run_cicero_cobinding.sbatch` now calls it instead of
  `export_tf_cicero_mm.py` (1 kb-bin path); added `PMAT`, `MIN_CELLS_PER_PEAK`,
  `MIN_PEAKS_PER_CELL`, and `IGNORE_FITCONNS=1` (force the pmat path on RBBP4 to
  compare vs gold). Validated the exporter end-to-end locally (valid mm, 100% barcode
  match, filters applied).
- **Open items / caveats:**
  - **pmat vs gold peak universe barely overlaps** (0 exact, 12% ≥1 bp, 7.8%
    center-in) — likely a **genome-build mismatch** (fitConns hg19 vs pmat hg38?) or
    the fitConns used a *separate* RBBP4-specific peak set. So a pmat run won't
    bit-for-bit reproduce fitConns; validate by network quality/significance, or get
    the colleague's exact inputs. **Next: liftOver test to settle the build question.**
  - Barcode-format match between `unified` mm and the pmat is asserted at runtime
    (fails loud) but unverified until a cluster run.
  - Cell-count parity unresolved: RBBP4 = 181,396 pmat cells vs a *back-calculated*
    (assumption-laden, possibly wrong) lab n≈8.5k. Not treating 8.5k as fact.
  - Non-fitConns path still keeps all edges at cluster time (`CLUSTER_QVAL=1`); wiring
    the new FDR into that filter is a deliberate follow-up.
- **Files:** `cobinding/scripts/export_pmat_tf_mm.R` (new),
  `cobinding/slurm/run_cicero_cobinding.sbatch`.

### 2026-09-23 — Cicero co-binding: add the missing significance (fitConns pval/qval)
- **Context:** Our co-binding cliques looked far less significant than the lab
  colleague's `fitConns` workbook. Root cause found in our own code: Cicero's
  `run_cicero()` emits **only** a `coaccess` score (regularized partial
  correlation, [-1,1]) — **no p-value / FDR**. `cicero_conns_to_edges.py`
  hardcoded `qval=0`, so downstream `q<=0.05` clique filtering was meaningless on
  any TF we ran ourselves. We only matched the reference on RBBP4 because we
  *inherited* the colleague's q-values from their fitConns file. The genuine
  Ren-lab Cicero papers (mouse cerebrum *Nature* 2021; human brain *Science* 2023)
  also just threshold the coaccess score — the per-pair significance is an add-on.
- **Diagnosis:** Reverse-engineered the fitConns `pval` from the reference RBBP4
  data — it's a **correlation-significance test on the coaccess score** (t-test /
  Fisher-z) with effective n ≈ number of cells/metacells (calibrated n≈8,531 for
  RBBP4, ~10k cells), then BH-adjusted. It's a smooth function of `coaccess`
  alone → rules out any permutation or distance-decay null.
- **Change:** `03_run_cicero.R` now records `n_metacell` in `cicero_info.tsv`;
  `cicero_conns_to_edges.py` computes a two-sided t-test (`df=n_eff-2`) on each
  pair's coaccess + BH `qval` (new `--effective-n`, default `metacell`, read from
  the sibling info file). `cluster_peaks.py` now filters on real FDR. Kept it in
  the co-binding path only — the Cicero **imputation** path is retired, untouched.
- **Outcome / caveat:** Model is exact at the q≈0.05 cutoff and the 2.22e-16 floor,
  within ~1 order of magnitude in the mid-range (reference likely uses a per-pair
  n we don't get from `run_cicero`'s output). Good enough for a first pass; refine
  to per-pair n if needed. **To try:** run our own Cicero on a TF (generates
  `n_metacell`), check the `significance: n_eff=… FDR<=0.05: X/Y` log line.
- **Files:** `Cicero/scripts/03_run_cicero.R`,
  `cobinding/scripts/cicero_conns_to_edges.py`.

### 2026-09-13 — Start of this research journal
- **Context:** Project has grown across many pipelines and several subtle,
  hard-won methods decisions; needed a single reviewable record of major updates.
- **Change:** Created `RESEARCH_JOURNAL.md` at the repo root, back-filled with the
  full history below.
- **Outcome:** Going forward, every major update gets an entry here.

### 2026-09-09 → 09-13 — ChromHMM-18 RPKM correlation: "all-red" was a color-scale artifact
- **Context:** Replicating the reference "TF chromHMM18 rpkm ratio pearson cor"
  figure. First cluster run looked degenerate — a near-uniform red heatmap
  (median off-diagonal Pearson r ≈ 0.96; `r(CTCF,RBBP4)=1.000`).
- **Investigation:** Detoured through "needs an across-TF ratio / log / Spearman"
  (all wrong for this data). The real issue: the meaningful structure lives in the
  0.65–0.90 band, and the original plot auto-scaled `vmin`≈0.68 **and** crammed in
  a 4th panel, saturating everything ≥0.85 to red.
- **Change:** Plain **RPKM → Pearson** is the correct and sufficient pipeline.
  Fixed the plot to the reference's scale (`vmin=0.65, vmax=1.0`, RdYlBu_r) and
  clean layout (dendrogram | heatmap | cell-count bars). With that, the real data
  reproduces the reference's two-block structure — CTCF/STAG2/E2F3/NEUROD4/RBBP4
  vs. MEF2A/THAP6/ZNF143 — faithfully.
- **Outcome:** RPKM normalizes state length + per-TF depth at once; no second
  normalization needed. Lesson: verify the *visualization* before re-deriving the
  method. Also applied the validated ChromHMM RPKM workflow to A/B compartments,
  and recorded why RPKM cannot rescue per-domain compartment correlation.
- **Files:** `tf_complex/scripts/{build_chromhmm_matrix,chromhmm_correlation}.py`,
  `tf_complex/slurm/04_chromhmm_rpkm.sbatch`. **Data:** ENCODE `ENCFF071AXS`
  (HEK293T ChromHMM-18, hg38). **Commits:** `5e8e3f0`, `0685fda`, `45356fa`,
  `afe584d`, `9f8ec27`.

### 2026-09-04 → 09-10 — Motif enrichment: scoring, backgrounds, and the accessibility wash
- **Context:** Can imputed data recover TF-specific motifs, and does an
  open-chromatin-region (OCR) background beat a whole-genome background?
- **Findings / changes:**
  - **AME `--scoring max` vs `avg`.** `max` maximized raw sensitivity but wrecked
    specificity (NRF1's own motif fell from rank 1 → rank 79) and produced
    meaningless 1e-100…1e-4000 p-values (AME's Fisher/rank-sum scales ~exponentially
    with sequence count). Reverted default to **`avg`**, which ranks the target
    motif at the top. The metric that matters is **target-motif rank**, not the
    absolute p-value. (Initial `avg→max` change was commit `7180509`; reverted
    `b220cf0`.)
  - **HOMER OCR background works correctly.** Against "open chromatin minus the
    TF's own peaks," the imputed TF motif comes out P≈1.0 — *real depletion*, not a
    bug: e.g. CTCF motif in only 2.3% of imputed target peaks vs 9.8% of the OCR
    background (real CTCF ChIP is >70%). The imputed peaks land in generic
    accessible regions and miss true sites, which stay in the background. Added
    `-cpg -olen 3` (HOMER practicalTips) to kill poly-G/Maz junk (`b365709`).
  - **Accessibility wash (core biological conclusion).** Imputed q<0.05 peaks are
    largely TF-nonspecific (75–91% shared across TFs) → generic GC motifs; a
    matched OCR background does **not** rescue TF specificity. Honest table:
    Raw recovers the motif; Imputed/genome-bg still recovers it (easy contrast);
    Imputed/OCR-bg fails for both AME(avg) and HOMER → imputation adds
    accessibility, not TF specificity.
  - Consensus subtraction (`TF_SPECIFIC=1`) partially recovers some motifs
    (NRF1 q≈0.025, HOXD10, FOXK1, HOXA5).
- **Files:** `downstream/slurm/motif_enrichment.sbatch`,
  `downstream/summarize_motif_table.py`, `downstream/emit_bins_bed.py`.
- **Memory:** `imputed-peaks-accessibility-wash.md`.

### 2026-09-04 — TF co-occupancy correlation confound (CPM / fraction is a Pearson no-op)
- **Context:** TF×TF co-occupancy matrix came out as an all-red "rich club";
  attempts to normalize by CPM / per-TF fraction didn't help.
- **Change / outcome:** Proved **per-TF (per-column) scaling — CPM, fraction — is a
  mathematical no-op for Pearson** (scale invariance; max|ΔR|≈2e-16). The confound
  is *structural* (peak-count / domain-size / domain-activity driven), not
  magnitude. Fixes that actually work: correlate over uniform units and normalize
  by both length and depth (→ this fed the ChromHMM-RPKM approach). Removed the
  broken `pca` metric (row-centering made sparse TFs collinear → spurious 36-TF
  block); defaulted to Pearson/Jaccard.
- **Commits:** `4b66def` (PCA bug), `fd811d6`, `8604de8` (CPM), `614afe6`
  (domain size), `b95e027` (fraction), `9970586` (bin width).
- **Memory:** `tf-complex-parallel-analysis.md`.

### 2026-08-26 → 09-01 — A/B compartments + Hi-C, and A-A/B-B co-binding
- **Context:** Stratify TF co-occupancy by A/B compartment to find complexes.
- **Change:** Built the Hi-C pipeline for A/B compartments (matched HEK293T Hi-C,
  SRR24709565, MboI, GSE233166 → hg38 via runHiC → cooltools), set resolution to
  **25 kb**, and added A-A and B-B correlations plus per-compartment motif analysis.
  Reformatted all correlation matrices to the "complexity" layout (heatmap +
  per-TF cell-count bar).
- **Commits:** `c4620a7`, `5459580`, `0d40691`, `ad4e496`, `5c2cba9`, `a7156fb`.
- **Memory:** `hic-compartments-cobinding.md`.

### 2026-08-26 → 09-09 — Co-binding higher-order clusters (Cicero) module
- **Context:** Discover higher-order co-binding regions beyond pairwise.
- **Change:** `cobinding/` module — per-TF Cicero peak co-accessibility → region
  graph → higher-order clusters/cliques (nodes are **peaks**, not TFs). Added
  single force-directed canvas, dropped isolated pairs, and annotated cliques with
  ChromHMM state, A/B compartment, and central genes (30.7% of distal partners sit
  in promoter chromatin).
- **Commits:** `c51d719`, `f8e9ea1`, `c6d9ada`, `46a63e6`, `b3ab734`, `db988eb`,
  `dcf5465`.
- **Memory:** `cobinding-higher-order-clusters.md`.

### 2026-08-19 → 08-25 — Motif Enrichment Analysis (MEA) pipeline introduced
- **Context:** First motif-enrichment capability on the imputed calls.
- **Change:** Added MEA scripts (HOMER known/de-novo + MEME-suite AME over
  HOCOMOCO v11 full), fixed region-shortage and background bugs, wired the
  genome-vs-OCR background comparison, and standardized peaks at MACS3 q<0.05.
- **Commits:** `1ca2f2b`, `acd8b4f`, `8899ec6`, `df9e7be`, `d51dceb`, `e48a65f`.

### 2026-08-17 — Downstream redesign: pos/neg universe, OCR filtering, peak-call standard
- **Context:** Earlier universes/peak definitions were inconsistent across TFs.
- **Change:** Redefined positive/negative universes, added Open-Chromatin-Region
  filtering, and set the project peak-calling standard. Corrected CTCF's positive
  peak set (an inflated ~322k-peak set from the wrong BED → ENCODE `ENCFF314ZAL`,
  HEK293 CTCF ChIP) and the CTCF negative-bin definition (not-CTCF-bound cCRE
  universe + zero-bulk-read filter). Committed configs for all 1000-TF imputations.
- **Commits:** `8f61059`, `dfad51a`, `a3b3770`, `d8d0ffa`, `9a2b847`.
- **Memory:** `ctcf-positive-peakset-fix.md`, `ctcf-negative-bin-definition.md`.

### 2026-07 → 2026-08-17 — Multi-TF downstream expansion + open-chromatin masking
- **Context:** Scale downstream from a few TFs to the panel; make imputed-only
  calls outside accessible chromatin less credible.
- **Change (detailed in `CHANGES_July_to_August_2026.md`):** `all_universe` eval
  mode and bulk×bulk reference matrices; HEK293T OmniATAC open-chromatin mask
  (GSE302716 via ENCODE ATAC pipeline v2.2.3; IDR-optimal ~72,182 peaks →
  ~57,286/3.03M bins "open"); post-mask peak recall and refreshed slides; large
  disk cleanup (~187 GB → ~11 GB) and `.gitignore` overhaul.
- **Note:** Most of this period was **not** committed (git quiet 07-01 → 08-17);
  that markdown is the durable record.

### 2026-06-04 → 06-29 — Unified imputer + peak-calling standardization; TF panel growth
- **Context:** Consolidate multiple imputation methods into one model and settle a
  reproducible peak-calling scheme.
- **Change:** Added the **unified** model; migrated MACS2 → **MACS3**; swept
  peak-calling q-value / span / per-bin quantile and coverage-spread parameters;
  tightened negative-background definition; added data-preparation pipelines and
  configs for new TFs with bulk data.
- **Commits:** `88b0e99`, `be1fc07`, `b558aed`, `641427a`, `e807…`(data-prep),
  `d4ab8f6`.

### 2026-05 — Imputation method bake-off
- **Context:** Find the best imputer for sparse per-TF binding.
- **Change:** Implemented and compared **cisTopic, FITS, scOpen, MAGIC, PUscOpen,
  scBasset, Borzoi**, plus **combined** pipelines (scBasset×cisTopic,
  scBasset×PUscOpen, scBasset_cisTopic_PUscOpen). Added Sens/Spec/F1 and
  held-out-reconstruction evals, bulk-comparison metrics, Cicero, and bigwig
  export. Repeatedly fixed data-leakage in the PU (positive-unlabeled) masking.
- **Commits:** many; e.g. `f28974e` (cisTopic), `911600a` (FITS), `db81a66`
  (scOpen), `102da25` (MAGIC), `a2e9757` (PUscOpen), `5d23e57` (scBasset),
  `7cc3a68` (Borzoi), `0d46f1f`/`fe7318d`/`224c91f` (leakage fixes).

### 2026-04-17 → 04-25 — Project bootstrap
- **Context:** Stand up the repo and the first imputation pipeline.
- **Change:** Initial import of the Paired-Tag pipeline, references, and conda env;
  renamed scope to **XuLab-Computations**; first cisTopic TF-imputation pipeline
  with data loaders, sparsity-tuned configs, and a held-out reconstruction eval.
- **Commits:** `b37d1a3`, `a5e2697`, `f28974e`, `86d7674`.

---

## Standing lessons (quick reference)
- **Cluster sync:** unchanged output usually = stale checkout on Longleaf; pull first.
- **Pearson scale-invariance:** per-TF CPM/fraction can't de-confound a Pearson
  correlation; fix the *units* (length + depth), not the magnitude.
- **Imputation = accessibility, not specificity:** imputed q<0.05 peaks are 75–91%
  shared across TFs; motif specificity survives only vs. easy backgrounds.
- **Motif ranking > motif p-value:** at large N every real motif is "1e-huge"; use
  the target's rank and its gap over the generic-GC pack.
- **Check the plot before the method:** the ChromHMM "all-red" scare was a `vmin`
  color-scale artifact, not a broken computation.
