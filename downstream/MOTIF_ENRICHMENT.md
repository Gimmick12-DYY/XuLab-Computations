# Motif enrichment on imputed calls

De novo (HOMER) + known-motif (AME/HOCOMOCO) enrichment on the **open-chromatin-
masked imputed** TF calls. One SLURM array task per TF; efficient by construction
(TFs run in parallel, HOMER threaded, steps skip if outputs exist).

## Pipeline

```
unified/work/<tf>/impute/matrix_csr.npz
        │  build_imputed_peaks.py  (bins bound in enough cells -> merged BED)
        ▼
downstream/motif/<tf>/<tf>.imputed_peaks.bed          (+ <tf>.background.bed)
        │
        ├── HOMER  findMotifsGenome.pl <bed> hg38 homer/ -useNewBg -p 8 -size given
        └── AME    bed2fasta <bed> hg38.fa -> ame(<fa>, HOCOMOCO.meme)
```

### 1. Called-peak definition (`build_imputed_peaks.py`)

The imputed matrix is binary (post open-mask + keep_raw). A bin is a "peak" when
imputed-bound in enough cells:

```
n_cells_bound = rowsum(matrix_csr)
keep bin iff n_cells_bound >= MIN_CELLS  AND  n_cells_bound/n_cells >= MIN_FRAC
             (optionally cap to the strongest TOP_N bins)
```

Adjacent kept bins are merged into intervals. **Sanity-check peak counts** against
the TF's bulk ChIP (tens of thousands) and tune `MIN_CELLS` / `MIN_FRAC` / `TOP_N`.

### 2. Background (accessibility-matched, recommended)

`build_imputed_peaks.py --open-bed <mask>` also writes `<tf>.background.bed`:
open-chromatin bins that are **not** called peaks (and not within `--flank-bp`).
Using this as the HOMER `-bg` / AME `--control` makes enriched motifs reflect TF
specificity rather than mere accessibility.

## Reference-gated by manifest

Motif enrichment only makes sense where the TF has a reference motif, so the run
is driven by `downstream/motif_reference_manifest.tsv`. The rule is fixed:

- **HOMER** runs for a TF when `homer = yes`.
- **AME** runs for a TF when `hocomoco_full` is present (HOCOMOCO v11 **full**).

A TF with only one reference runs only that step; a TF with neither is skipped.
Keep the manifest in sync with your availability spreadsheet and extend it with
every study TF (incl. the panel: ctcf, maz, …).

## Run

```bash
# 1. see eligible TFs (HOMER-yes OR HOCOMOCO-full) and get the array size
LIST=1 bash downstream/slurm/motif_enrichment.sbatch
N=$(LIST=1 bash downstream/slurm/motif_enrichment.sbatch | grep -vc '^#')

# 2. submit; each TF runs only the pipeline(s) it has a reference for
sbatch --array=0-$((N-1)) downstream/slurm/motif_enrichment.sbatch

# explicit subset (still gated by the manifest)
TFS="mef2a znf143" sbatch --array=0-1 downstream/slurm/motif_enrichment.sbatch
```

### Key env knobs (all optional; defaults mirror the provided samples)

| Var | Default | Meaning |
|-----|---------|---------|
| `MANIFEST` | `downstream/motif_reference_manifest.tsv` | per-TF reference availability |
| `TFS` | manifest-eligible | explicit TF list (still gated by the manifest) |
| `CONDA_ENV` | `unified` | env with numpy+scipy for the BED builder |
| `HOMER_MODULE` / `MEME_MODULE` | unset | `module load` args for HOMER / MEME |
| `HOMER_GENOME` | `hg38` | HOMER genome name or `/path/hg38.fa` |
| `HG38_FA` | `downstream/cache/hg38.fa` | FASTA for bed2fasta |
| `HOCOMOCO_MEME` | `cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme` | AME motif DB (HOCOMOCO v11 full) |
| `OPEN_BED` | OmniATAC IDR peaks | mask used to build the background (swap for `HEK293T_ATAC_union3.bed.gz`) |
| `MIN_CELLS` / `MIN_FRAC` / `TOP_N` | `2` / `0.02` / `0` | peak thresholds |
| `HOMER_BG` | `0` | `1` -> HOMER `-bg background.bed` instead of `-useNewBg` |
| `AME_CONTROL` | `0` | `1` -> AME `--control background.fa` instead of shuffled |

## Notes

- Defaults reproduce the provided commands exactly (`-useNewBg`, AME shuffled
  control). Set `HOMER_BG=1` / `AME_CONTROL=1` for the accessibility-matched
  background (recommended for open-chromatin-restricted imputed calls).
- HOMER preparses the genome on first use. For large parallel arrays, warm it by
  running one TF first, or set a shared `-preparsedDir`, to avoid concurrent
  preparse writes.
- Outputs under `downstream/motif/` are gitignored (regenerable); the scripts are
  tracked.
