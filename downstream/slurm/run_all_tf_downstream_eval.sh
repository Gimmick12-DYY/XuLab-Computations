#!/bin/bash
# -----------------------------------------------------------------------------
# run_all_tf_downstream_eval.sh
#
# Submit AUROC (compare_pos_neg), bin coverage (bin_sensitivity_specificity),
# and MACS3 peak coverage (peak_coverage) for CTCF + every TF x benchmark mode.
#
# Per TF the matrix inputs are:
#   raw     = unified/work/<tf>/mm
#   unified = unified/work/<tf>/impute
#
# Benchmark bins come from downstream/bins_<tf>_<mode>/pos_neg_bins.tsv
# (built by run_all_tf_benchmark_modes.sh).
#
# Peak coverage: MACS3 is run once per TF (all_bound mode). fixed_n and
# peak_cutoff reuse those peaks via REUSE_PEAKS_FROM and only recompute
# coverage/AUROC against the new pos/neg labels.
#
# Usage:
#   bash downstream/slurm/run_all_tf_downstream_eval.sh
#   DRY_RUN=1 bash downstream/slurm/run_all_tf_downstream_eval.sh
#   bash downstream/slurm/run_all_tf_downstream_eval.sh MAZ ZNF282
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
SLURM_DIR="${DOWN_DIR}/slurm"

DEFAULT_TFS=(CTCF ZBTB7A ZIC2 MAZ ZNF777 NFYA ZNF282)
MODES=(all_bound fixed_n peak_cutoff peak_cutoff_q05)

if [[ "${1:-}" == "all" || $# -eq 0 ]]; then
  TFS=("${DEFAULT_TFS[@]}")
elif [[ $# -gt 0 ]]; then
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
  if [[ ! -f "${mm}/regions.tsv.gz" ]]; then
    echo "[skip] ${TF}: missing ${mm}/regions.tsv.gz" >&2
    continue
  fi
  if [[ ! -d "${impute}" ]]; then
    echo "[skip] ${TF}: missing impute dir ${impute}" >&2
    continue
  fi
  inputs="raw=${mm} unified=${impute}"
  peak_base="${DOWN_DIR}/peak_coverage_${tf}_all_bound"
  peak_parent=""

  for mode in "${MODES[@]}"; do
    bins="${DOWN_DIR}/bins_${tf}_${mode}/pos_neg_bins.tsv"
    if [[ ! -f "${bins}" ]]; then
      echo "[skip] ${TF} ${mode}: missing ${bins}" >&2
      continue
    fi

    tag="${tf}_${mode}"
    cmp_out="${DOWN_DIR}/compare_pos_neg_${tag}"
    bin_out="${DOWN_DIR}/bin_coverage_${tag}"
    peak_out="${DOWN_DIR}/peak_coverage_${tag}"

    echo "========== ${TF} / ${mode} =========="

    sbatch_or_echo --job-name="ds_auroc_${tag}" \
      --export=ALL,BINS_TSV="${bins}",OUT_DIR="${cmp_out}",OUT_NAME="compare_pos_neg_${tag}",INPUTS="${inputs}" \
      "${SLURM_DIR}/compare_pos_neg.sbatch"

    sbatch_or_echo --job-name="ds_bincov_${tag}" \
      --export=ALL,BINS_TSV="${bins}",OUT_DIR="${bin_out}",OUT_NAME="bin_coverage_${tag}",INPUTS="${inputs}" \
      "${SLURM_DIR}/bin_sensitivity_specificity.sbatch"

    if [[ "${mode}" == "all_bound" ]]; then
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run] sbatch peak_coverage ${tag} (parent for reuse)"
        peak_parent="DRYRUN"
      else
        peak_parent="$(sbatch --parsable --job-name="ds_peakcov_${tag}" \
          --export=ALL,BINS_TSV="${bins}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}" \
          "${SLURM_DIR}/peak_coverage.sbatch")"
        echo "[submit] peak_coverage ${tag} job=${peak_parent} (MACS parent for reuse)"
      fi
    else
      dep=()
      [[ -n "${peak_parent}" && "${peak_parent}" != "DRYRUN" ]] && dep=(--dependency="afterok:${peak_parent}")
      sbatch_or_echo "${dep[@]}" --job-name="ds_peakcov_${tag}" \
        --export=ALL,BINS_TSV="${bins}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}",REUSE_PEAKS_FROM="${peak_base}" \
        "${SLURM_DIR}/peak_coverage.sbatch"
    fi
  done
done

cat <<EOF

Queued (or dry-run printed) up to $(( ${#TFS[@]} * ${#MODES[@]} * 3 )) jobs.

Key outputs per TF/mode (<tag> = <tf>_<mode>):
  AUROC + ECDF:     downstream/compare_pos_neg_<tag>/compare_pos_neg_<tag>.json
  Bin coverage:     downstream/bin_coverage_<tag>/bin_coverage_<tag>.json
  Peak coverage:    downstream/peak_coverage_<tag>/peak_coverage_<tag>_table.txt

Monitor:
  squeue -u \$USER -n ds_auroc_,ds_bincov_,ds_peakcov_

Collect metrics table (all TFs incl. CTCF):
  python3 downstream/collect_tf_eval_summary.py > downstream/tf_eval_summary.tsv

Each TF (including CTCF) uses mode=fixed_n|all_bound|peak_cutoff and raw + unified only.

EOF
