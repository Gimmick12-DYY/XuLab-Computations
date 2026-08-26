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
        └── AME    bed2fasta <bed> hg38.fa -> ame(<peaks.fa>, HOCOMOCO.meme)
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

### 2. TF-specific (consensus-subtracted) peaks — `TF_SPECIFIC=1`

The plain peaks above are dominated by accessibility (all TFs' pseudobulks are
~identical, 75–91% peak overlap → generic GC/CpG motifs). `TF_SPECIFIC=1` calls
peaks on the **observed-minus-expected residual** instead — count-preserving so
the residual stays peaky and MACS3 can call it:

```
consensus  = mean over TFs of (pseudobulk / sum)          # accessibility SHAPE, sums to ~1
expected_X = consensus * total_X                          # counts under pure accessibility
residual_X = max(0, signal_X - scale * expected_X)        # scale=CONSENSUS_SCALE (default 1)
```

Bins where TF X exceeds its accessibility expectation stay sharp (count units);
proportional-to-accessibility washes to ~0. Peaks are then called on that residual
(same MACS3 q<0.05).

> **Note:** the earlier `rank` normalization made residuals *uniform*, so MACS3
> called **0 peaks** for every TF. The default is now `fraction` (count-like).
> If too many TFs still fall under `MIN_PEAKS`, soften with `CONSENSUS_SCALE=0.7`
> (subtract less accessibility) before concluding there is no TF-specific signal.

Build the consensus once (default `--normalize fraction`):

```bash
python downstream/build_consensus_accessibility.py \
  --out-npy downstream/motif/consensus_accessibility.npy      # all impute dirs, rank-normalized
```

Outputs go to a **separate tree** (`downstream/motif_tfspec/`) so they never mix
with the plain run. If a TF has little signal above consensus it falls under
`MIN_PEAKS` and is skipped — that is the honest "no TF-specific signal" outcome.

### 3. Background (accessibility-matched, opt-in)

`--open-bed <mask>` also writes `<tf>.background.bed`: open-chromatin bins that are
**not** covered by a called peak (and not within `--flank-bp`). Available as HOMER
`-bg` / AME `--control` via `HOMER_BG=1` / `AME_CONTROL=1` (default off — see Notes).

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

# TF-specific (consensus-subtracted) peaks -> motif_tfspec/
python downstream/build_consensus_accessibility.py --out-npy downstream/motif/consensus_accessibility.npy
TF_SPECIFIC=1 sbatch --array=0-$((N-1)) downstream/slurm/motif_enrichment.sbatch
```

### Key env knobs (all optional)

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
| `TF_SPECIFIC` | `0` | `1` -> consensus-subtracted peaks (needs `CONSENSUS_NPY`); output tree `motif_tfspec/` |
| `CONSENSUS_NPY` | `motif/consensus_accessibility.npy` | consensus built by `build_consensus_accessibility.py` (rebuild with default `--normalize fraction`) |
| `CONSENSUS_SCALE` | `1.0` | subtract `scale`×expected accessibility; `<1` softer / more peaks |
| `HOMER_BG` | `0` | `0` -> HOMER `-useNewBg` (original); `1` -> `-bg background.bed` (accessibility-matched) |
| `AME_CONTROL` | `0` | `0` -> AME shuffled (original); `1` -> `--control background.fa` (accessibility-matched) |

## Notes

- **Defaults match the original commands exactly**: HOMER `-useNewBg -p 8 -size given`,
  AME shuffled control (`HOMER_BG=0`, `AME_CONTROL=0`).
- The accessibility-matched background (`HOMER_BG=1` / `AME_CONTROL=1`, using
  `<tf>.background.bed` = open chromatin minus this TF's peaks) was tried to counter
  the fact that open-chromatin peaks are GC/CpG-rich and enrich for generic GC-box/CpG
  motifs (SP2/AP2/MECP2). It did **not** recover TF-specific motifs — because the
  imputed peaks themselves are ~accessibility (75–91% shared across TFs), so there is
  little TF-specific signal for any background to isolate. Kept as an opt-in only.
- HOMER preparses the genome on first use. For large parallel arrays, warm it by
  running one TF first, or set a shared `-preparsedDir`, to avoid concurrent
  preparse writes.
- Outputs under `downstream/motif/` are gitignored (regenerable); the scripts are
  tracked.
