# shellcheck shell=bash
# Shared paths / env for the HEK293T Hi-C pipeline. Source after `set -euo pipefail`.
# Every value is overridable from the environment.

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
HIC="${HIC:-${XULAB}/hic}"

# runHiC data tree (-p points here). Large/machine-local (gitignored).
DATA="${HIC_DATA:-${HIC}/data}"          # data/hg38/, data/HiC-SRA/
WORK="${HIC_WORK:-${HIC}/work}"          # runHiC workspace: datasets.tsv + output folders
GENOME="${GENOME:-hg38}"
GENOME_DIR="${DATA}/${GENOME}"
GENOME_FA="${GENOME_FA:-${GENOME_DIR}/${GENOME}.fa}"
CHROMSIZES="${CHROMSIZES:-${GENOME_DIR}/${GENOME}.chrom.sizes}"

# Source FASTA to build the (main-chrom) reference from -- reuse the existing cache.
SRC_GENOME_FA="${SRC_GENOME_FA:-${XULAB}/downstream/cache/hg38.fa}"
MAIN_CHROMS="${MAIN_CHROMS:-chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY chrM}"

SAMPLES_TSV="${SAMPLES_TSV:-${HIC}/metadata/datasets.tsv}"
SRR="${SRR:-SRR24709565}"                # HEK293T Hi-C (GSM7405563, GSE233166)
ENZYME="${ENZYME:-MboI}"                 # in-situ Hi-C, Rao 2014 (GATC 4-cutter)
ALIGNER="${ALIGNER:-bwa-mem}"            # bwa-mem (standard) | chromap (fast, recommended for depth)
CHUNK_SIZE="${CHUNK_SIZE:-1500000}"      # read pairs per mapping chunk (runHiC --chunkSize)

# Compartment calling (cooltools eigs-cis).
COMPARTMENT_RES="${COMPARTMENT_RES:-25000}"    # 25 kb bins for A/B compartments (deep library)

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

CONDA_ENV="${CONDA_ENV:-/work/users/d/y/dyy12/conda/envs/hic}"
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  CONDA_ENV="/nas/longleaf/home/dyy12/.conda/envs/hic"
fi
export PATH="${CONDA_ENV}/bin:${PATH}"
