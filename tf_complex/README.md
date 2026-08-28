# TF co-occupancy → candidate protein complexes

Standalone pipeline (Image-#11 checklist): **A/B compartments → correlate TFs by
co-occupancy → shared motif → protein complex**. Independent of the Cicero
higher-order co-binding module (`cobinding/`) — this is a parallel analysis of the
whole TF panel.

## Idea
TFs that bind the **same genomic regions** are candidate members of a protein
complex. We correlate every TF pair by their raw binding profile, stratify by A/B
compartment (from the matched Hi-C), cluster the correlation into TF groups, then
(next steps) test whether each group shares a motif → call it a complex.

## Why RAW binding, not imputed
The imputed matrices are **open-chromatin masked**, which forces every TF into the
same open bins and would inflate co-occupancy. So profiles come from the **raw**
per-TF pseudobulk (`unified/work/<tf>/mm`), sum across cells per 1 kb bin.

## Pipeline

```bash
mamba env create -f tf_complex/environment.yml -p /work/users/d/y/dyy12/conda/envs/tf_complex

# needs the Hi-C A/B calls first (hic step 05): compartments_25000.AB.bed
sbatch tf_complex/slurm/run_tf_correlation.sbatch
#   METRIC=jaccard CORR_THRESHOLD=0.4 sbatch tf_complex/slurm/run_tf_correlation.sbatch
#   REBUILD=0 sbatch tf_complex/slurm/run_tf_correlation.sbatch     # reuse the matrix
```

### 1. `build_tf_matrix.py` — raw binding matrix (run once)
Every TF's raw pseudobulk (`mm/matrix.mtx.gz` summed over cells) → `work/tf_matrix.npz`
(bins × TFs, CSC) + `tfs.txt` + `regions.tsv`.

### 2. `tf_correlation.py` — co-occupancy → complexes

**Sparsity + coverage both matter.** Raw scCUT&Tag is sparse and each TF is profiled
in *different* cells, so at 1 kb two co-binding TFs rarely hit the same bin (Spearman
≈ 0, flat heatmap); and TFs vary ~250× in depth, so "bound = any signal" makes
broad/deep TFs overlap trivially (a coverage rich-club). So the tool:
- **aggregates** fine bins into `AGG_BP` windows (default 10 kb) so co-localizing TFs
  share a bin;
- defines **"bound" = each TF's top-`TOP_N` strongest bins** (default 20 000) —
  coverage-equalized and background-free, so overlap reflects *specific* co-binding;
- defaults to **`cooccur`** = `log2(observed/expected)` overlap, with the expected
  computed over the **full scope universe** (genome / all-A / all-B bins) — a
  co-occurrence *enrichment* ("bind together above chance"). `spearman/pearson/jaccard`
  also available;
- prints **coverage + off-diagonal magnitude diagnostics** so a flat/rich-club result
  is explained.

Outputs (three scopes — **genome / A-only / B-only**):
- `cooccur_{genome,A,B}.tsv` — co-occurrence enrichment (always written).
- `corr_{...}.tsv` — the chosen `METRIC` matrix (if not cooccur).
- **`candidate_complexes.tsv`** — TFs joined into **connected components** where the
  genome score ≥ `CLUSTER_THRESHOLD` (complex_id, n_tfs, mean_intra_score, members).
- **`tf_compartment.tsv`** — per-TF A/B preference, size-normalized
  (`log2(signal_fracA / exp_fracA)`).
- `correlation_heatmap.png` (`--plot`), clustered.

## Knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `WORK_ROOT` | `unified/work` | per-TF raw `mm` matrices |
| `AB_BED` | `hic/work/compartments/compartments_25000.AB.bed` | A/B calls (hic step 05) |
| `REBUILD` | `1` | `0` = reuse the built matrix |
| `METRIC` | `cooccur` | `cooccur` (log2 obs/exp) \| `spearman` \| `pearson` \| `jaccard` |
| `AGG_BP` | `10000` | aggregate 1 kb bins into this window before correlating |
| `TOP_N` | `20000` | "bound" = each TF's top-N strongest bins (coverage-equalize, drop background) |
| `MIN_TFS_PER_BIN` | `2` | (spearman/pearson only) feature bins must be bound in ≥ this many TFs |
| `CLUSTER_THRESHOLD` | `1.0` | complex if score ≥ this (cooccur log2 fold; 1.0 = 2× co-binding) |

## Reading the result
- **`candidate_complexes.tsv`** = TF groups that co-occupy → protein-complex candidates.
- Compare **A-only vs B-only** correlations to see whether a complex is
  compartment-specific.
- **`tf_compartment.tsv`** gives each TF's A/B lean (size-normalized).

Next in the checklist: for each candidate complex, test **shared/same motif**
(HOMER/AME on the group's binding), then **define the protein complex**.

## Tracking
`tf_complex/work/` (binding matrix) is gitignored; `results/` TSVs are small and
kept. Scripts / slurm / env / README tracked.
