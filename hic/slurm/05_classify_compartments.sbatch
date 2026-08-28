#!/bin/bash
#SBATCH --job-name=hic_ab_classify
#SBATCH --output=hic_ab_classify-%j.out
#SBATCH --error=hic_ab_classify-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#
# Classify 25 kb bins into A/B compartments + compartment sizes (the normalizer),
# then optionally assign a region set (co-binding modules / TF peaks) to A/B with
# size-normalized enrichment. This is the A/B-compartment co-binding step.
#
#   sbatch hic/slurm/05_classify_compartments.sbatch
#   REGIONS=/work/.../cobinding/results/RBBP4/nodes.tsv LABEL=RBBP4_cobind \
#     sbatch hic/slurm/05_classify_compartments.sbatch
#
# Prereq: 04_compartments produced compartments_<res>.E1.bedGraph.

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

CONDA_ENV_HIC="${CONDA_ENV:-hic}"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV_HIC}" 2>/dev/null || true

COMP="${WORK}/compartments"
E1="${E1_BEDGRAPH:-${COMP}/compartments_${COMPARTMENT_RES}.E1.bedGraph}"
PREFIX="${COMP}/compartments_${COMPARTMENT_RES}"
[[ -s "${E1}" ]] || { echo "missing E1 bedGraph: ${E1} (run 04_compartments)" >&2; exit 1; }

echo "[$(date)] classify bins -> A/B + sizes"
python "${HIC}/scripts/classify_compartments.py" \
  --e1-bedgraph "${E1}" --out-prefix "${PREFIX}"

# Optional: assign a region set to A/B, size-normalized.
if [[ -n "${REGIONS:-}" ]]; then
  [[ -s "${REGIONS}" ]] || { echo "REGIONS not found: ${REGIONS}" >&2; exit 1; }
  LABEL="${LABEL:-$(basename "${REGIONS}" | sed 's/\.[^.]*$//')}"
  OUT="${COMP}/${LABEL}.compartments.tsv"
  echo "[$(date)] assign regions ${LABEL} -> A/B (size-normalized)"
  python "${HIC}/scripts/assign_regions_to_compartments.py" \
    --regions "${REGIONS}" --ab-bed "${PREFIX}.AB.bed" \
    --out "${OUT}" --summary "${COMP}/${LABEL}.enrichment.tsv" --label "${LABEL}"
fi
echo "[$(date)] done -> ${COMP}"
