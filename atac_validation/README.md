# ATAC-seq validation of the HEK293T open-chromatin mask

Independent wild-type HEK293T ATAC-seq datasets used to **validate and extend**
the OmniATAC open-chromatin mask (`data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz`,
GSE302716) that gates the unified imputer.

Three things this module does:

1. **Download** two independent WT HEK293T ATAC-seq datasets.
2. **Overlap** each new mask with the other and with the current OmniATAC mask
   (bin-level Jaccard + directional recovery).
3. **Union** all three into one open-chromatin mask and check the downstream
   effect (per-TF open-positive recovery), producing a drop-in `open_chromatin_bed`.

Scripts are written to run on Longleaf; nothing here is run locally.

---

## Datasets (`metadata/samples.tsv`)

| Dataset tag | GEO | Samples used | Layout | Why |
|-------------|-----|--------------|--------|-----|
| `gse283384_wt` | GSE283384 | 2 lipofectamine-only **controls** (GSM8661240/41) | PAIRED | Cleanest WT HEK293T; MORC2-OE samples excluded |
| `gse152177_wt` | GSE152177 | 8 libraries (4 G/G + 4 A/A at rs4420550), **pooled** | PAIRED | WT HEK293T; single-SNP locus, treated as reps |

SRR accessions are hard-coded in `metadata/samples.tsv` (resolved from GEO/SRA).
`keep=yes` rows are downloaded and pooled per dataset.

---

## Pipeline

All paths are overridable via env vars in `scripts/lib.sh` (defaults assume the
Longleaf checkout at `/work/users/d/y/dyy12/XuLab`).

```bash
# 0. env (once)
mamba env create -f atac_validation/environment.yml \
  -p /work/users/d/y/dyy12/conda/envs/atac

# 0b. bowtie2 index from the shared hg38 FASTA (once)
sbatch atac_validation/scripts/build_bowtie2_index.sh

# 1. download FASTQ (10 keep=yes runs -> array 0..9)
N=$(awk -F'\t' '$1 !~ /^#/ && $6=="yes"' atac_validation/metadata/samples.tsv | wc -l)
sbatch --array=0-$((N-1)) atac_validation/slurm/00_download.sbatch

# 2. trim -> bowtie2 -> filter/dedup -> pool -> MACS2 (one task per dataset)
sbatch --array=0-1 atac_validation/slurm/01_align_call_peaks.sbatch
#   -> data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz
#   -> data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz
# If pooled BAMs already exist but peaks are missing, resume with:
#   sbatch --array=0-1 atac_validation/slurm/01b_macs_from_pooled.sbatch

# 3. overlap (bin Jaccard) + union mask
sbatch atac_validation/slurm/02_overlap_union.sbatch
#   -> results/mask_sizes.tsv, mask_overlap.tsv, mask_jaccard_heatmap.png
#   -> data/HEK293T_ATAC_union3.bed.gz   (union open_chromatin_bed)
#   -> results/union_summary.tsv

# 4. downstream check: per-TF open-positive recovery under each mask + union
sbatch atac_validation/slurm/03_downstream_recovery.sbatch
#   -> results/open_positive_recovery_all_universe.tsv
```

### Alignment / peak-calling choices (`01_align_call_peaks.sbatch`)

- **Trim:** fastp, auto adapter detection (`--detect_adapter_for_pe`).
- **Align:** bowtie2 `--very-sensitive -X 2000` against hg38.
- **Filter:** proper pairs, `MAPQ >= 30`, `-F 1804` (drop unmapped/secondary/dup/
  supplementary), duplicates removed (`samtools markdup -r`), autosomes + chrX/Y
  only (drops chrM/scaffolds/alts).
- **Peaks:** MACS2 `-f BAMPE -g hs -q 0.01` on the pooled BAM, then ENCODE hg38
  blacklist subtracted if `BLACKLIST_BED` exists.

This is a lightweight, self-contained pipeline (not the ENCODE Caper/Cromwell
pipeline that produced the OmniATAC IDR mask). Concordance is compared at 1 kb
**bin** resolution, which is robust to the difference in peak callers.

---

## Using the union mask downstream

`data/HEK293T_ATAC_union3.bed.gz` is a plain BED of merged open intervals — a
drop-in for `open_chromatin_bed`. To re-mask imputes with the union:

```yaml
# unified/configs/<tf>.yaml (or default.yaml)
open_chromatin_bed: /work/users/d/y/dyy12/XuLab/data/HEK293T_ATAC_union3.bed.gz
open_chromatin_mask_imputed_only: true
```

then re-apply post-hoc (no retrain) with the existing:

```bash
TFS="ctcf,maz,zic2,zbtb7a,znf777,nfya,znf282" \
  sbatch --array=0-6 unified/slurm/02b_apply_open_chromatin_mask.sbatch
```

Overlap logic reuses `unified/scripts/_regions.py`, so "open bin" here means
exactly what it means to the imputer.

---

## Outputs

| Path | Tracked | Contents |
|------|---------|----------|
| `results/mask_sizes.tsv` | yes | open-bin count per mask |
| `results/mask_overlap.tsv` | yes | pairwise intersection / union / Jaccard / recovery |
| `results/mask_jaccard_heatmap.png` | no (`*.png` ignored) | 3×3 Jaccard heatmap |
| `results/union_summary.tsv` | yes | union sizes, gains over OmniATAC, per-source unique bins |
| `results/open_positive_recovery_*.tsv` | yes | per-TF open-positive % under each mask |
| `data/HEK293T_ATAC_<dataset>_MACS2.narrowPeak.gz` | no (`data/` ignored) | validation peaks |
| `data/HEK293T_ATAC_union3.bed.gz` | no (`data/` ignored) | union mask (drop-in) |
| `work/` | no | FASTQ / BAM / MACS2 intermediates |

Tracked in git: `scripts/`, `slurm/`, `metadata/`, `environment.yml`, `results/*.tsv`,
this README.
