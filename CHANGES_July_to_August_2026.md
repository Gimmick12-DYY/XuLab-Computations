# Changes: July → mid-August 2026

Documentation of work done in `XuLab/` from early July through 17 August 2026.
Most of this period was **not** committed to git (last commits on 2026-07-01);
this note is the durable record of pipelines, eval modes, open-chromatin work,
and cleanup.

**Panel TFs (main slide / bulk analyses):** CTCF, MAZ, ZIC2, ZBTB7A, ZNF777, NFYA, ZNF282.

---

## 1. Timeline (high level)

| When | Theme |
|------|--------|
| Early July | Multi-TF downstream expansion; peak-cutoff / fixed-n / final slide defs |
| Mid–late July | Presentable tables; frag-per-cell summaries; vs-bulk Spearman/Pearson experiments |
| ~28 Jul – 6 Aug | `all_universe` eval mode; bulk×bulk reference matrices |
| ~6 Aug | HEK293T OmniATAC (ENCODE Path A); open-chromatin mask on unified impute |
| ~8 Aug | Post-mask peak recall; refreshed `slide_all_universe` + `slide_top12k_q05` |
| ~13 Aug | Open-pos universe slides; open-restricted vs-bulk matrices |
| ~17 Aug | Downstream disk cleanup (~187 GB → ~11 GB); `.gitignore` overhaul |

---

## 2. Unified imputer — open-chromatin masking

### Motivation

New imputed calls outside accessible chromatin are biologically less credible.
Masking them (while **keeping observed raw** everywhere) reduces false imputed-only
signal before downstream AUROC / peak recall.

### Open-chromatin definition (how regions were found)

1. **Source:** HEK293T OmniATAC-seq, GEO **GSE302716**  
   Replicates: SRR34546238, SRR34546237, SRR34546232, SRR34546231.
2. **Pipeline:** Official ENCODE ATAC-seq pipeline **v2.2.3** via Caper/Cromwell + Apptainer  
   (`XuLab/atac_omniatac/`; workflow `70bfc7e8-3760-496f-bd8e-5a4bc9e02b51`).
3. **Peak set used:** IDR **optimal** peaks  
   `idr.optimal_peak.narrowPeak.gz` (~**72,182** peaks).
4. **Repo symlink:**  
   `data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz`
5. **1 kb bin mask:** any unified bin overlapping those peaks is “open”  
   (~**57,286 / ~3.03M** bins).

Details and runbook: `atac_omniatac/README.md`.

### Code / config changes

| Piece | Role |
|-------|------|
| `unified/scripts/_regions.py` | `open_chromatin_bin_mask()` |
| `unified/scripts/02_impute.py` | Mask **before** `keep_raw` union |
| `unified/scripts/02b_apply_open_chromatin_mask.py` | Post-hoc remask of existing CSRs (no retrain) |
| `unified/configs/default.yaml` | `open_chromatin_bed`, `open_chromatin_mask_imputed_only: true` |
| `unified/slurm/02b_apply_open_chromatin_mask.sbatch` | Array job over `unified/work/*/impute` |

**Semantics (default):**

- Delete **new imputed** entries outside open chromatin.
- Preserve **raw** observations everywhere (`keep_raw: true` + `open_chromatin_mask_imputed_only: true`).
- Disable with `open_chromatin_bed: null` (or `"off"`).

**Applied post-hoc** to the **7-TF panel** (backups: `matrix_csr.npz.pre_openmask`).  
Other ~70+ TF imputes were not necessarily remasked.

**Effect on CTCF all_universe Bin.pos:** only ~12.4% of CTCF all-universe positives overlap open chromatin; gains on closed positives were wiped back toward raw. On **open** positives, unified Bin.pos stayed ~80.7%.

---

## 3. Downstream evaluation modes

Slide / AUROC / bin / peak metrics evolved through several positive/negative definitions:

| Mode tag | Positives | Negatives | Status |
|----------|-----------|-----------|--------|
| `fixed_n` | Fixed-size pos/neg | Size-matched | Superseded; dirs deleted Aug 17 |
| `peak_cutoff` | Peak-cutoff bins | Matched neg | Superseded; deleted |
| `peak_cutoff_q05` | q05 peak cutoff | Matched | Kept (feeds `slide_top12k_q05` for some TFs) |
| `final` | Final curated set | — | Kept (CTCF/MAZ/ZIC2 for top12k) |
| `all_bound` | All TF-bound cCRE | Size-matched random | Superseded; deleted |
| **`all_universe`** | All TF-bound cCRE | **All** zero-bulk unbound candidates | **Current**; openmask peaks recalled |
| **`all_universe_openpos`** | `all_bound ∩ open` | Same full `all_candidates` | **Current** (Aug 13) |

### Key builders / runners

- `downstream/slurm/run_all_universe_panel.sh`
- `downstream/build_all_universe_bins_from_cache.py` (offline bins from bedcov cache)
- `downstream/build_slide_all_universe.py` → `slide_tables/slide_all_universe.{txt,csv}`
- `downstream/build_all_universe_openpos_bins.py`
- `downstream/build_slide_all_universe_openpos.py` → `slide_all_universe_openpos.{txt,csv}`
- `downstream/build_slide_top12k_q05.py` → `slide_top12k_q05.{txt,csv}`

### SLURM export fix

`--export=VAR=value` broke when `INPUTS="raw=… unified=…"` contained spaces.  
Runners were fixed to export **variable names** (`--export=ALL,BINS_TSV,OUT_DIR,…`) so paths with spaces/equals survive.

### Snapshot results (post-openmask)

**Full all_universe** (`slide_all_universe.txt`, Aug 8) — CTCF unified AUROC ~0.62, Bin.pos ~36% (many positives outside open).

**Open-pos** (`slide_all_universe_openpos.txt`, Aug 13) — same negatives; positives restricted to open:

| TF | n_pos | unified AUROC | Bin.pos % | Peak.pos % | ΔAUROC |
|----|------:|--------------:|----------:|-----------:|-------:|
| CTCF | 39871 | 0.879 | 80.7 | 60.3 | +0.112 |
| MAZ | 27175 | 0.959 | 92.4 | 87.8 | +0.189 |
| ZIC2 | 20317 | 0.967 | 94.6 | 75.6 | +0.197 |
| ZBTB7A | 7006 | 0.993 | 98.9 | 92.4 | +0.105 |
| ZNF777 | 5494 | 0.985 | 97.4 | 90.8 | +0.265 |
| NFYA | 7361 | 0.999 | 100 | 99.3 | +0.020 |
| ZNF282 | 4977 | 0.985 | 97.9 | 75.6 | +0.245 |

---

## 4. Bulk concordance matrices (raw / imputed × bulk)

Scripts under `downstream/`:

- `plot_vs_bulk_spearman_heatmap.py` (+ `slurm/plot_vs_bulk_spearman.sbatch`)
- `plot_vs_bulk_qn_pearson_heatmap.py` (QN then Pearson; Spearman is QN-invariant)
- Related: `plot_raw_vs_imputed_spearman_heatmap.py`, `plot_sc_vs_bulk_validation.py`

**Feature spaces:** `all` (~3M bins), `union_peaks` (~458k), `ccre`; optional **`--open-chromatin`** intersects with OmniATAC open bins → output tags `*_open_*`.

**Outputs:** `downstream/plots/tf_bulk_pearson/`  
e.g. `spearman_union_peaks_{raw,imputed,bulk}_vs_bulk.*`,  
`spearman_{union_peaks,all}_open_*_vs_bulk.*`

**Findings (summary):**

- Bulk×bulk on `union_peaks` shows strong shared bulk correlation (explains washed diagonals).
- Regenerating after openmask (Aug 13) did **not** produce a clean diagonal-dominant imputed×bulk matrix.
- Restricting correlation bins to open chromatin (`*_open_*`) also did not fully recover specificity.
- Track-level / matrix-level QN experiments reduced shared wash but also flattened the diagonal.

---

## 5. Downstream disk cleanup (17 Aug 2026)

`downstream/` went from ~**187 GB** → ~**11 GB**.

### Deleted (safe / unused)

1. Old multi-method peak trees & sweeps (~85 GB):  
   bare `peak_coverage_{ctcf,maz,…}`, `*_qsweep`, `*_opsweep`, `*_12kpos*`, `*_old12k`, `peak_coverage_setdb1`, bare `bins_*`, `bins/`, `pos_neg_all10`, etc.
2. Superseded modes (~42 GB): `*_all_bound`, `*_fixed_n`, `*_peak_cutoff` (non-q05) for peak/bin/compare trees.
3. All `compare_pos_neg_*` (~340 MB) — AUROC already baked into slide tables.
4. All `bin_coverage_*` (~2 MB).
5. All `peak_coverage_*/**/synthetic.bed` (~57 GB MACS intermediates); kept small tables + called peaks.

### Kept

- Eval dirs for: `all_universe`, `all_universe_openpos`, `final`, `peak_cutoff_q05`  
  (peaks/tables without `synthetic.bed`)
- `bins_*` for those modes, `slide_tables/`, `plots/`, scripts + `slurm/`, `cache/`

---

## 6. `.gitignore` overhaul

Replaced one-off paths with regenerable-output patterns so eval dumps stop cluttering git status:

```
downstream/cache/
downstream/bins/
downstream/bins_*/
downstream/bin_coverage_*/
downstream/peak_coverage_*/
downstream/compare_pos_neg_*/
downstream/slide_tables/
downstream/plots/
downstream/tracks/
…
atac_omniatac/
unified/work/
```

**Still intended to be tracked:** Python/R scripts, `downstream/slurm/`, configs, READMEs, this changelog.

---

## 7. Where to look now

| Need | Path |
|------|------|
| Current full-universe slides | `downstream/slide_tables/slide_all_universe.{txt,csv}` |
| Open-positive slides | `downstream/slide_tables/slide_all_universe_openpos.{txt,csv}` |
| Top-12k / q05 slides | `downstream/slide_tables/slide_top12k_q05.{txt,csv}` |
| Open chromatin peaks | `data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz` |
| OmniATAC pipeline | `atac_omniatac/` |
| Impute + mask docs | `unified/README.md` |
| Vs-bulk heatmaps | `downstream/plots/tf_bulk_pearson/` |
| Pre-mask impute backups (panel) | `unified/work/<tf>/impute/matrix_csr.npz.pre_openmask` |

### Regenerating deleted evals

- AUROC / bin coverage: re-run `compare_pos_neg.sbatch` / `bin_sensitivity_specificity.sbatch` with `bins_<tf>_<mode>/pos_neg_bins.tsv`.
- Peak coverage: `peak_coverage.sbatch` (rebuilds `synthetic.bed` + MACS); openpos can `REUSE_PEAKS_FROM` all_universe peaks when the matrix is unchanged.
- Slide tables: `build_slide_all_universe.py`, `build_slide_all_universe_openpos.py`, `build_slide_top12k_q05.py`.

---

## 8. Note on version control

Git history for this window is sparse (`2026-07-01` commits only). Treat this file plus `unified/README.md` and `atac_omniatac/README.md` as the primary narrative of July–August changes until corresponding commits are made.
