# HEK293T Hi-C → A/B compartments

Process the matched HEK293T **Hi-C** to **hg38** from raw reads (there was only an
hg18 matrix), then call **A/B compartments** — the input for the planned
co-binding analysis. Uses **runHiC** ([XiaoTaoWang/HiC_pipeline](https://github.com/XiaoTaoWang/HiC_pipeline))
for mapping→filtering→binning(+ICE), then **cooltools** for compartments.

## Dataset

| | |
|---|---|
| Run | **SRR24709565** (paired 2×150, ~330M read pairs, deep) |
| Sample | GSM7405563 — "HEK293T_1000cells_2", WT, in-situ Hi-C (**GSE233166**) |
| Method | Rao et al. 2014 in-situ Hi-C → enzyme **MboI** (GATC) |

This is the HEK293T Hi-C matched to the 1000-cell scCUT&Tag TF panel.

> **Confirm the enzyme.** The GEO protocol cites Rao 2014 (MboI). If the full
> protocol says DpnII (isoschizomer, same GATC — filtering is equivalent) or
> Arima, update `metadata/datasets.tsv` (4th column) accordingly.

## Pipeline

All paths overridable via `scripts/lib.sh`. Defaults assume the Longleaf checkout
at `/work/users/d/y/dyy12/XuLab`.

```bash
# 0. env (once)
mamba env create -f hic/environment.yml -p /work/users/d/y/dyy12/conda/envs/hic

# 1. download the SRA run -> data/HiC-SRA/SRR24709565.sra
sbatch hic/slurm/00_download.sbatch

# 2. build hg38 (main chromosomes) + aligner index -> data/hg38/
sbatch hic/slurm/01_prepare_genome.sbatch
#   ALIGNER=chromap sbatch hic/slurm/01_prepare_genome.sbatch   # if using chromap

# 3. runHiC pileup: map -> filter -> bin(+ICE)  [HEAVY: 330M pairs]
sbatch hic/slurm/02_pileup.sbatch
#   ALIGNER=chromap sbatch hic/slurm/02_pileup.sbatch           # much faster

# 4. QC
sbatch hic/slurm/03_quality.sbatch

# 5. A/B compartments (GC-phased) @ 25 kb -> E1 eigenvector track
sbatch hic/slurm/04_compartments.sbatch
#   COMPARTMENT_RES=100000 sbatch hic/slurm/04_compartments.sbatch

# 6. classify bins -> A/B + compartment sizes, and assign a region set (size-normalized)
sbatch hic/slurm/05_classify_compartments.sbatch
COBIND_TF=RBBP4 sbatch hic/slurm/05_classify_compartments.sbatch   # peaks + modules
REGIONS=/work/.../cobinding/results/RBBP4/nodes.tsv LABEL=RBBP4_peaks \
  sbatch hic/slurm/05_classify_compartments.sbatch
```

### `datasets.tsv` (runHiC metadata)

Tab-separated, no header: `SRA_prefix  cell_line  bio_replicate  enzyme`
```
SRR24709565	HEK293T	R1	MboI
```

### runHiC layout

runHiC reads a `../data/` tree and writes outputs into the workspace (`hic/work/`):
```
hic/data/
├── hg38/        hg38.fa (main chroms) + .fai + .chrom.sizes + bwa/chromap index
└── HiC-SRA/     SRR24709565.sra
hic/work/        datasets.tsv, alignments-hg38/, pairs-hg38/, filtered-hg38/,
                 coolers-hg38/*.mcool, runHiC-*.log, compartments/
```

### Aligner choice

`bwa-mem` (default) is the standard but slow at 330M pairs. **`ALIGNER=chromap`**
is far faster and fine for compartment-scale analysis — recommended here. Rerun
`01_prepare_genome.sbatch` with the matching `ALIGNER` so the right index exists.

## A/B compartments → co-binding

`04_compartments.sbatch` runs `cooltools eigs-cis` at **25 kb** (`COMPARTMENT_RES`,
default 25000) on the balanced matrix with a **GC phasing track**, so the first
eigenvector E1 is oriented **E1>0 = A** (open, GC-rich) / **E1<0 = B**. If the
`.mcool` lacks a 25 kb level, the step coarsens+balances one from a divisor
resolution automatically. Outputs under `work/compartments/`:

- `compartments_25000.cis.vecs.tsv` — per-bin E1/E2/E3
- `compartments_25000.E1.bedGraph` — A/B track (chrom start end E1)

**Step 6 — classify + size-normalize** (`05_classify_compartments.sbatch`):
- `classify_compartments.py` → `*.AB.bed` (per-bin A/B), `*.domains.bed` (merged
  compartment runs), `*.sizes.tsv` (**compartment sizes** = total A vs B bp, the
  normalizer).
- `assign_regions_to_compartments.py` (general; takes a BED or any chr:s-e file —
  co-binding `nodes.tsv`/`modules.tsv` or per-TF peaks) → per-region A/B call +
  **size-normalized enrichment**: `log2(obs_fracA / exp_fracA)` where `exp_fracA`
  is the A share of the genome, plus a binomial test. So a bigger compartment
  doesn't trivially win.

This is the A/B-compartment **co-binding** analysis: run it on RBBP4's co-binding
modules and on each TF's peaks to get their A/B preference, then feed the
correlation / motif / complex steps.

## Outputs & tracking

`hic/data/` and `hic/work/` are gitignored (large, regenerable). Tracked:
`scripts/`, `slurm/`, `metadata/datasets.tsv`, `environment.yml`, this README.

## Notes

- **Compute:** step 2 is the bottleneck (330M pairs). With `bwa-mem` budget days;
  with `chromap` hours. SLURM time is set to 4 days for bwa-mem — lower it for chromap.
- **`runHiC quality` flags** have varied across releases; if `03` errors, check
  `runHiC quality -h` and adjust.
- **Genome:** main chromosomes only (no alt/scaffold) so multi-mapping doesn't
  distort contacts or compartments.
- **25 kb compartments:** finer than the usual 100 kb–1 Mb, but this library is
  deep (~330M pairs) so eigs at 25 kb are supported. If E1 looks noisy on
  sparser chromosomes, bump `COMPARTMENT_RES` (e.g. 50000/100000).
