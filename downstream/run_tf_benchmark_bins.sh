#!/bin/bash
# -----------------------------------------------------------------------------
# run_tf_benchmark_bins.sh
#
# Build pos/neg bin sets for each benchmark TF under three positive definitions:
#   all_bound   - all TF-bound cCRE bins
#   fixed_n     - top 5000 TF-bound cCRE bins by peak signal
#   peak_cutoff - TF-bound cCRE bins from bulk peaks with q<0.01 & p<1e-5
#
# True negatives (all modes): exclude TF-bound cCREs, bedcov zero-read filter,
# random sample matched to #positives.
#
# SETDB1 is excluded from the default TF list.
#
# Usage:
#   bash downstream/run_tf_benchmark_bins.sh
#   bash downstream/run_tf_benchmark_bins.sh MAZ ZNF282
#   SUBMIT=1 bash downstream/run_tf_benchmark_bins.sh all
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DOWN_DIR="${ROOT_DIR}/downstream"
SBATCH_SCRIPT="${DOWN_DIR}/slurm/prepare_pos_neg_bins.sbatch"

declare -A MM_DIR=(
  [ZBTB7A]="${ROOT_DIR}/unified/work/zbtb7a/mm"
  [ZIC2]="${ROOT_DIR}/unified/work/zic2/mm"
  [MAZ]="${ROOT_DIR}/unified/work/maz/mm"
  [ZNF777]="${ROOT_DIR}/unified/work/znf777/mm"
  [NFYA]="${ROOT_DIR}/unified/work/nfya/mm"
  [ZNF282]="${ROOT_DIR}/unified/work/znf282/mm"
)

DEFAULT_TFS=(ZBTB7A ZIC2 MAZ ZNF777 NFYA ZNF282)
MODES=(all_bound fixed_n peak_cutoff)

if [[ "${1:-}" == "all" ]]; then
  TFS=("${DEFAULT_TFS[@]}")
elif [[ $# -gt 0 ]]; then
  TFS=("$@")
else
  TFS=("${DEFAULT_TFS[@]}")
fi

submit_job() {
  local tf="$1" mode="$2" mm="$3" peaks="$4" bam="$5" out="$6"
  local n_pos=0
  [[ "${mode}" == "fixed_n" ]] && n_pos=5000
  local exports="ALL,MM_DIR=${mm},OUT_DIR=${out},TF_PEAKS_BED=${peaks},BULK_BAM=${bam}"
  exports+=",POS_MODE=${mode},N_POS=${n_pos},NEG_MAX_COVERAGE=0,NEGATIVE_MODE=random"
  exports+=",EXCLUDE_ALL_CCRE=0,PEAK_Q_MAX=0.01,PEAK_P_MAX=1e-5"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    echo "[submit] ${tf} ${mode} -> ${out}"
    sbatch --export="${exports}" "${SBATCH_SCRIPT}"
  else
    echo "[dry-run] sbatch --export=${exports} ${SBATCH_SCRIPT}"
  fi
}

for TF in "${TFS[@]}"; do
  [[ "${TF}" == "SETDB1" ]] && { echo "[skip] ${TF} excluded from benchmark"; continue; }
  mm="${MM_DIR[$TF]:-}"
  if [[ -z "${mm}" || ! -d "${mm}" ]]; then
    echo "[skip] ${TF}: MM_DIR not found (${mm:-unset})" >&2
    continue
  fi
  line="$(awk -v tf="${TF}" '$1==tf {print $3; exit}' "${ROOT_DIR}/data_prep/fetch_tf_bulk.sh" 2>/dev/null || true)"
  # Resolve peaks/bam from data dir naming convention.
  peaks="$(ls "${DATA_DIR}/${TF}"_*_peaks.bed 2>/dev/null | head -1 || true)"
  bam="$(ls "${DATA_DIR}/${TF}"_*.bam 2>/dev/null | head -1 || true)"
  if [[ -z "${peaks}" || -z "${bam}" ]]; then
    echo "[skip] ${TF}: missing peaks or BAM under ${DATA_DIR}" >&2
    continue
  fi
  for mode in "${MODES[@]}"; do
    out="${DOWN_DIR}/bins_${TF,,}_${mode}"
    submit_job "${TF}" "${mode}" "${mm}" "${peaks}" "${bam}" "${out}"
  done
done

echo "Done. Set SUBMIT=1 to queue jobs."
