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
#   REGIONS=/work/.../cobinding/results/RBBP4/nodes.tsv LABEL=RBBP4_peaks \
#     sbatch hic/slurm/05_classify_compartments.sbatch
#   COBIND_TF=RBBP4 sbatch hic/slurm/05_classify_compartments.sbatch
#     # -> peaks (nodes.tsv) + module spans (modules.tsv --bounding-box)
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

assign_regions() {
  local regions="$1" label="$2"; shift 2
  local out="${COMP}/${label}.compartments.tsv"
  echo "[$(date)] assign regions ${label} -> A/B (size-normalized)"
  python "${HIC}/scripts/assign_regions_to_compartments.py" \
    --regions "${regions}" --ab-bed "${PREFIX}.AB.bed" \
    --out "${out}" --summary "${COMP}/${label}.enrichment.tsv" --label "${label}" "$@"
}

# Explicit region set (BED or chr:s-e file).
if [[ -n "${REGIONS:-}" ]]; then
  [[ -s "${REGIONS}" ]] || { echo "REGIONS not found: ${REGIONS}" >&2; exit 1; }
  LABEL="${LABEL:-$(basename "${REGIONS}" | sed 's/\.[^.]*$//')}"
  assign_regions "${REGIONS}" "${LABEL}"
fi

# Co-binding TF: peaks (nodes.tsv) + module genomic spans (modules.tsv).
if [[ -n "${COBIND_TF:-}" ]]; then
  COB_RES="${COB_RES:-${XULAB}/cobinding/results/${COBIND_TF}}"
  NODES="${COB_RES}/nodes.tsv"
  MODULES="${COB_RES}/modules.tsv"
  [[ -s "${NODES}" ]] || { echo "COBIND_TF=${COBIND_TF} but missing ${NODES} (run cobinding first)" >&2; exit 1; }
  assign_regions "${NODES}" "${COBIND_TF}_peaks"
  if [[ -s "${MODULES}" ]]; then
    assign_regions "${MODULES}" "${COBIND_TF}_modules" --bounding-box
  else
    echo "[$(date)] no ${MODULES}; skip module-span assignment"
  fi
fi
echo "[$(date)] done -> ${COMP}"
