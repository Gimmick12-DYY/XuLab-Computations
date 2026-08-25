# Motif enrichment on imputed calls

De novo (HOMER) + known-motif (AME/HOCOMOCO) enrichment on the **open-chromatin-
masked imputed** TF calls. One SLURM array task per TF; efficient by construction
(TFs run in parallel, HOMER threaded, steps skip if outputs exist).

## Pipeline

```
unified/work/<tf>/impute/matrix_csr.npz
        │  call_imputed_peaks.py  (pseudobulk -> MACS3 q<0.05 -> narrowPeak)
        ▼
downstream/motif/<tf>/<tf>.imputed_peaks.bed          (+ <tf>.background.bed)
        │
        ├── HOMER  findMotifsGenome.pl <bed> hg38 homer/ -useNewBg -p 8 -size given
        └── AME    bed2fasta <bed> hg38.fa -> ame(<fa>, HOCOMOCO.meme)
```

### 1. Called-peak definition (`call_imputed_peaks.py`)

Peaks are called at **MACS3 q < 0.05** on the imputed pseudobulk, using the exact
method in `peak_coverage.py` (the project peak-calling standard): per-bin signal =
sum across cells → synthetic fragment BED → `macs3 callpeak --nomodel --shift
-extsize/2 --extsize 200 -q 0.05 --keep-dup all --nolambda`, with the effective
genome size set from the track's own background quantile. This is uniform and
statistically comparable across TFs (unlike a fixed prevalence cutoff, which gave
1–10k peaks depending only on imputation density).

If MACS3 calls fewer than `MIN_PEAKS` (default 200), the builder exits 3 and the
TF is **skipped** (logged to `motif/skipped_insufficient_peaks.txt`) — a TF with
that little confident imputed signal can't yield a valid enrichment.

### 2. Background (accessibility-matched, recommended)

`--open-bed <mask>` also writes `<tf>.background.bed`: open-chromatin bins that are
**not** covered by a called peak (and not within `--flank-bp`). Using this as the
HOMER `-bg` / AME `--control` makes enriched motifs reflect TF specificity rather
than mere accessibility.

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
| `CONDA_ENV` | `data_prep` | env with macs3 + numpy/scipy (peak calling) |
| `HOMER_MODULE` / `MEME_MODULE` | unset | `module load` args for HOMER / MEME |
| `HOMER_GENOME` | `hg38` | HOMER genome name or `/path/hg38.fa` |
| `HG38_FA` | `downstream/cache/hg38.fa` | FASTA for bed2fasta |
| `HOCOMOCO_MEME` | `cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme` | AME motif DB (HOCOMOCO v11 full) |
| `OPEN_BED` | `data/HEK293T_ATAC_union3.bed.gz` | open-chromatin mask for accessibility-matched background (same union used to remask imputes) |
| `MACS_Q` | `0.05` | MACS3 q-value cutoff for imputed peak calling |
| `MIN_PEAKS` | `200` | skip a TF if MACS3 calls fewer than this many peaks |
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
