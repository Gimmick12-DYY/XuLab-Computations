#!/bin/bash
#SBATCH --job-name=ame_cdbk
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#
# AME enrichment vs Codebook top1 for the 11 previously-unknown TFs that have a
# Codebook PWM. Reuses motif_enrichment.sbatch (peaks reuse, SKIP_HOMER).
#
#   sbatch --array=0-10 motif_analysis/slurm/motif_enrichment_codebook.sbatch
#   BG_MODE=ocr sbatch --array=0-10 motif_analysis/slurm/motif_enrichment_codebook.sbatch
#   REGION=bins BG_MODE=genome sbatch --array=0-10 motif_analysis/slurm/motif_enrichment_codebook.sbatch
#   REGION=bins BG_MODE=ocr sbatch --array=0-10 motif_analysis/slurm/motif_enrichment_codebook.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
MOTIF_DIR="${MOTIF_DIR:-${ROOT_DIR}/motif_analysis}"

export ALL_TFS=1
export SKIP_HOMER=1
export TFS="${TFS:-znf507 mbd3 znf703 glyr1 flywch1 znf367 znf746 pogk zbtb40 znf606 znf503}"
export HOCOMOCO_MEME="${HOCOMOCO_MEME:-${MOTIF_DIR}/cache/motifdb/codebook_top1.meme}"
export BG_MODE="${BG_MODE:-genome}"
export REGION="${REGION:-peaks}"

echo "[codebook-ame] TFS=${TFS}"
echo "[codebook-ame] BG_MODE=${BG_MODE} REGION=${REGION}"
echo "[codebook-ame] MEME_DB=${HOCOMOCO_MEME}"

# Re-exec the main enrichment script as this array task (same SLURM_ARRAY_TASK_ID).
exec bash "${MOTIF_DIR}/slurm/motif_enrichment.sbatch"
