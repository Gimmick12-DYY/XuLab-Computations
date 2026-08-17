#!/bin/bash
# -----------------------------------------------------------------------------
# run_peak_cutoff_q05.sh
#
# 1) Submit peak_cutoff bin-prep with PEAK_Q_MAX=0.05 for all (or listed) TFs.
#    Outputs: downstream/bins_<tf>_peak_cutoff_q05/
# 2) Chain AUROC / bin coverage / peak coverage eval jobs that wait on bins.
#    Peak calling reuses all_bound MACS peaks (REUSE_PEAKS_FROM).
#
# Usage:
#   bash downstream/slurm/run_peak_cutoff_q05.sh
#   bash downstream/slurm/run_peak_cutoff_q05.sh MAZ ZNF282
#   SKIP_EVAL=1 bash downstream/slurm/run_peak_cutoff_q05.sh   # bins only
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DOWN_DIR="${ROOT_DIR}/downstream"
SLURM_DIR="${DOWN_DIR}/slurm"
SBATCH_BINS="${SLURM_DIR}/prepare_pos_neg_bins.sbatch"
DEFAULT_TFS=(CTCF ZBTB7A ZIC2 MAZ ZNF777 NFYA ZNF282)
MODE=peak_cutoff_q05
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ $# -eq 0 ]]; then
  TFS=("${DEFAULT_TFS[@]}")
else
  TFS=("$@")
fi

for TF in "${TFS[@]}"; do
  tf="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
  LINE=HEK293
  [[ "${TF}" == "NFYA" ]] && LINE=K562
  if [[ "${TF}" == "CTCF" ]]; then
    PEAKS="${DATA_DIR}/CTCF_${LINE}_peaks.bed"
    if [[ -f "${DATA_DIR}/ENCFF139DCW.bam" ]]; then
      BULK="${DATA_DIR}/ENCFF139DCW.bam"
    else
      BULK="${DATA_DIR}/CTCF_${LINE}.bam"
    fi
  else
    PEAKS="${DATA_DIR}/${TF}_${LINE}_peaks.bed"
    BULK="${DATA_DIR}/${TF}_${LINE}.bam"
  fi
  MM="${ROOT_DIR}/unified/work/${tf}/mm"
  IMPUTE="${ROOT_DIR}/unified/work/${tf}/impute"
  OUT="${DOWN_DIR}/bins_${tf}_${MODE}"
  for f in "${PEAKS}" "${BULK}" "${MM}/regions.tsv.gz"; do
    if [[ ! -e "${f}" ]]; then
      echo "ERROR: missing ${f}" >&2
      exit 1
    fi
  done
  if [[ "${SKIP_EVAL}" != "1" && ! -d "${IMPUTE}" ]]; then
    echo "ERROR: missing impute dir ${IMPUTE}" >&2
    exit 1
  fi

  echo "[submit] ${TF} POS_MODE=peak_cutoff PEAK_Q_MAX=0.05 -> ${OUT}"
  bin_job="$(sbatch --parsable --time=04:00:00 --job-name="ds_bins_${tf}_q05" --export=ALL,\
MM_DIR="${MM}",\
OUT_DIR="${OUT}",\
TF_PEAKS_BED="${PEAKS}",\
BULK_BAM="${BULK}",\
POS_MODE=peak_cutoff,\
PEAK_Q_MAX=0.05,\
PEAK_P_MAX=1e-5,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED=2026 \
    "${SBATCH_BINS}")"
  echo "[submit] bins job=${bin_job}"

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    continue
  fi

  tag="${tf}_${MODE}"
  bins_tsv="${OUT}/pos_neg_bins.tsv"
  inputs="raw=${MM} unified=${IMPUTE}"
  peak_base="${DOWN_DIR}/peak_coverage_${tf}_all_bound"
  cmp_out="${DOWN_DIR}/compare_pos_neg_${tag}"
  bin_out="${DOWN_DIR}/bin_coverage_${tag}"
  peak_out="${DOWN_DIR}/peak_coverage_${tag}"

  auroc_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
    --job-name="ds_auroc_${tag}" \
    --export=ALL,BINS_TSV="${bins_tsv}",OUT_DIR="${cmp_out}",OUT_NAME="compare_pos_neg_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/compare_pos_neg.sbatch")"
  echo "[submit] auroc job=${auroc_job} (afterok:${bin_job})"

  bincov_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
    --job-name="ds_bincov_${tag}" \
    --export=ALL,BINS_TSV="${bins_tsv}",OUT_DIR="${bin_out}",OUT_NAME="bin_coverage_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/bin_sensitivity_specificity.sbatch")"
  echo "[submit] bincov job=${bincov_job} (afterok:${bin_job})"

  if [[ ! -d "${peak_base}" ]]; then
    echo "[warn] ${TF}: missing ${peak_base}; peak coverage will call MACS de novo" >&2
    peak_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
      --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV="${bins_tsv}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}" \
      "${SLURM_DIR}/peak_coverage.sbatch")"
  else
    peak_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
      --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV="${bins_tsv}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}",REUSE_PEAKS_FROM="${peak_base}" \
      "${SLURM_DIR}/peak_coverage.sbatch")"
  fi
  echo "[submit] peakcov job=${peak_job} (afterok:${bin_job})"
done

cat <<EOF

Queued peak_cutoff_q05 bins (+ eval unless SKIP_EVAL=1).

Bin counts land in:
  downstream/bins_<tf>_peak_cutoff_q05/pos_neg_bins.meta.json  (n_pos, n_neg)

After jobs finish:
  python3 downstream/collect_tf_eval_summary.py > downstream/tf_eval_summary.tsv

Monitor:
  squeue -u \$USER -n ds_bins_,ds_auroc_,ds_bincov_,ds_peakcov_

EOF
