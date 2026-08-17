#!/bin/bash
# -----------------------------------------------------------------------------
# run_final_downstream_eval.sh
#
# AUROC / bin coverage / peak coverage for the final pos/neg definition
# (downstream/bins_<tf>_final/).
#
# Usage:
#   bash downstream/slurm/run_final_downstream_eval.sh
#   DRY_RUN=1 bash downstream/slurm/run_final_downstream_eval.sh
#   bash downstream/slurm/run_final_downstream_eval.sh MAZ ZIC2
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
SLURM_DIR="${DOWN_DIR}/slurm"
MODE=final

DEFAULT_TFS=(CTCF ZBTB7A ZIC2 MAZ ZNF777 NFYA ZNF282)

if [[ "${1:-}" == "all" || $# -eq 0 ]]; then
  TFS=("${DEFAULT_TFS[@]}")
else
  TFS=("$@")
fi

sbatch_or_echo() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] sbatch $*"
  else
    sbatch "$@"
  fi
}

for TF in "${TFS[@]}"; do
  tf="${TF,,}"
  mm="${ROOT_DIR}/unified/work/${tf}/mm"
  impute="${ROOT_DIR}/unified/work/${tf}/impute"
  bins="${DOWN_DIR}/bins_${tf}_${MODE}/pos_neg_bins.tsv"

  if [[ ! -f "${mm}/regions.tsv.gz" ]]; then
    echo "[skip] ${TF}: missing ${mm}/regions.tsv.gz" >&2
    continue
  fi
  if [[ ! -d "${impute}" ]]; then
    echo "[skip] ${TF}: missing impute dir ${impute}" >&2
    continue
  fi
  if [[ ! -f "${bins}" ]]; then
    echo "[skip] ${TF}: missing ${bins} (run run_final_benchmark_bins.sh first)" >&2
    continue
  fi

  inputs="raw=${mm} unified=${impute}"
  tag="${tf}_${MODE}"
  cmp_out="${DOWN_DIR}/compare_pos_neg_${tag}"
  bin_out="${DOWN_DIR}/bin_coverage_${tag}"
  peak_out="${DOWN_DIR}/peak_coverage_${tag}"

  # Prefer all_bound MACS peaks when available; CTCF SCREEN bins fall back to
  # legacy peak_coverage_ctcf or all_bound peaks.
  reuse=""
  if [[ -d "${DOWN_DIR}/peak_coverage_${tf}_all_bound" ]]; then
    reuse="${DOWN_DIR}/peak_coverage_${tf}_all_bound"
  elif [[ "${tf}" == "ctcf" && -d "${DOWN_DIR}/peak_coverage_ctcf" ]]; then
    reuse="${DOWN_DIR}/peak_coverage_ctcf"
  fi

  echo "========== ${TF} / ${MODE} =========="

  sbatch_or_echo --job-name="ds_auroc_${tag}" \
    --export=ALL,BINS_TSV="${bins}",OUT_DIR="${cmp_out}",OUT_NAME="compare_pos_neg_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/compare_pos_neg.sbatch"

  sbatch_or_echo --job-name="ds_bincov_${tag}" \
    --export=ALL,BINS_TSV="${bins}",OUT_DIR="${bin_out}",OUT_NAME="bin_coverage_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/bin_sensitivity_specificity.sbatch"

  if [[ -n "${reuse}" ]]; then
    sbatch_or_echo --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV="${bins}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}",REUSE_PEAKS_FROM="${reuse}" \
      "${SLURM_DIR}/peak_coverage.sbatch"
  else
    sbatch_or_echo --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV="${bins}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}" \
      "${SLURM_DIR}/peak_coverage.sbatch"
  fi
done

cat <<EOF

Queued final-mode eval for: ${TFS[*]}

Collect when done:
  python3 downstream/collect_tf_eval_summary.py > downstream/tf_eval_summary.tsv

EOF
