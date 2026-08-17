# shellcheck shell=bash
# Shared paths / environment for atac_validation SLURM scripts.
# Source this after `set -euo pipefail`. Every value is overridable from the
# environment so paths can move without editing the scripts.

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
AV="${AV:-${XULAB}/atac_validation}"

# Large, machine-local outputs (gitignored).
WORK="${AV_WORK:-${AV}/work}"           # fastq/, <dataset>/bam/, <dataset>/peaks/
REF_DIR="${AV_REF:-${AV}/ref}"          # bowtie2 index
RESULTS="${AV_RESULTS:-${AV}/results}"  # small TSV/BED overlap + union outputs (tracked)

# Reference genome (reuse scBasset/unified hg38 cache).
GENOME_FA="${GENOME_FA:-${XULAB}/downstream/cache/hg38.fa}"
BT2_INDEX="${BT2_INDEX:-${REF_DIR}/hg38}"

# ENCODE hg38 blacklist. Optional: if the file is missing the filter is skipped
# (with a warning) rather than failing the run.
BLACKLIST_BED="${BLACKLIST_BED:-${XULAB}/downstream/cache/hg38.blacklist.bed.gz}"

# 1 kb bin universe shared by every unified impute (any TF works; CTCF is canonical).
REGIONS_TSV="${REGIONS_TSV:-${XULAB}/unified/work/ctcf/impute/regions.tsv}"

# The current open-chromatin reference we are validating against.
OMNIATAC_PEAKS="${OMNIATAC_PEAKS:-${XULAB}/data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz}"

SAMPLES_TSV="${SAMPLES_TSV:-${AV}/metadata/samples.tsv}"

# Autosomes + chrX/Y (drops chrM, scaffolds, alt contigs before peak calling).
MAIN_CHROMS="${MAIN_CHROMS:-chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY}"

# MACS2 stringency for the pooled per-dataset call.
MACS_Q="${MACS_Q:-0.01}"

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

# conda env carrying sra-tools, fastp, bowtie2, samtools, macs2, bedtools,
# numpy/pandas/matplotlib (see atac_validation/environment.yml).
CONDA_ENV="${CONDA_ENV:-/work/users/d/y/dyy12/conda/envs/atac}"
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  CONDA_ENV="/nas/longleaf/home/dyy12/.conda/envs/atac"
fi
PY="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

# Rows of samples.tsv with keep=yes for a given dataset -> stdout as "gsm\tsrr".
# Usage: dataset_runs gse283384_wt
dataset_runs() {
  local ds="$1"
  awk -F'\t' -v ds="${ds}" '
    $1 !~ /^#/ && $1==ds && $6=="yes" { print $2"\t"$3 }
  ' "${SAMPLES_TSV}"
}

# All keep=yes runs across datasets -> stdout as "dataset\tgsm\tsrr" (for the
# download array). Order is stable (file order) so array indices are reproducible.
all_runs() {
  awk -F'\t' '$1 !~ /^#/ && $6=="yes" { print $1"\t"$2"\t"$3 }' "${SAMPLES_TSV}"
}
