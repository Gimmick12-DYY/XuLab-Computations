#!/bin/bash
# -----------------------------------------------------------------------------
# run_znf282_downstream.sh
#
# After unified impute exists for ZNF282, wire GEO bulk (GSM2466515) into the
# final-mode benchmark:
#   1. ensure peaks BED (liftOver hg19→hg38) exists
#   2. submit / reuse BAM align if missing
#   3. prepare all_bound final bins (after BAM)
#   4. run AUROC / bin / peak coverage vs raw+unified (after bins)
#
# Usage (from repo root, as dyy12):
#   bash downstream/slurm/run_znf282_downstream.sh
#   DRY_RUN=1 bash downstream/slurm/run_znf282_downstream.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"
DOWN_DIR="${ROOT_DIR}/downstream"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
SLURM_DIR="${DOWN_DIR}/slurm"
TF=ZNF282
tf=znf282
MODE=final

sbatch_or_echo() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] sbatch $*"
    echo "DRYRUN_JOB"
  else
    sbatch --parsable "$@"
  fi
}

# Peaks (fast, local)
bash "${ROOT_DIR}/data_prep/fetch_znf282_bulk.sh"

PEAKS="${DATA_DIR}/${TF}_HEK293_peaks.bed"
BAM="${DATA_DIR}/${TF}_HEK293.bam"
MM="${ROOT_DIR}/unified/work/${tf}/mm"
IMPUTE="${ROOT_DIR}/unified/work/${tf}/impute"
BINS_OUT="${DOWN_DIR}/bins_${tf}_${MODE}"
BINS_TSV="${BINS_OUT}/pos_neg_bins.tsv"

for f in "${PEAKS}" "${MM}/regions.tsv.gz"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
[[ -d "${IMPUTE}" ]] || { echo "ERROR: missing impute ${IMPUTE}" >&2; exit 1; }

BAM_DEP=()
if [[ -f "${BAM}" && -f "${BAM}.bai" ]]; then
  echo "[znf282] BAM present: ${BAM}"
else
  EXISTING="$(squeue -u "${USER}" -n znf282_bulk_bam -h -o '%i' 2>/dev/null | head -1 || true)"
  if [[ -n "${EXISTING}" ]]; then
    echo "[znf282] BAM align already queued/running: job ${EXISTING}"
    BAM_DEP=(--dependency="afterok:${EXISTING}")
  else
    echo "[znf282] submitting BAM align"
    ALIGN_JOB=$(sbatch_or_echo "${ROOT_DIR}/data_prep/slurm/align_znf282_bulk.sbatch")
    echo "[znf282] align job=${ALIGN_JOB}"
    [[ "${ALIGN_JOB}" != "DRYRUN_JOB" ]] && BAM_DEP=(--dependency="afterok:${ALIGN_JOB}")
  fi
fi

BINS_DEP=("${BAM_DEP[@]}")
if [[ -f "${BINS_TSV}" ]]; then
  echo "[znf282] bins present: ${BINS_TSV}"
  BINS_JOB=""
else
  echo "[znf282] submitting final all_bound bins -> ${BINS_OUT}"
  BINS_JOB=$(sbatch_or_echo "${BINS_DEP[@]}" --job-name="ds_bins_${tf}_${MODE}" \
    --export=ALL,\
MM_DIR="${MM}",\
OUT_DIR="${BINS_OUT}",\
TF_PEAKS_BED="${PEAKS}",\
BULK_BAM="${BAM}",\
POS_MODE=all_bound,\
N_POS=0,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED=2026 \
    "${SLURM_DIR}/prepare_pos_neg_bins.sbatch")
  echo "[znf282] bins job=${BINS_JOB}"
fi

EVAL_DEP=()
if [[ -n "${BINS_JOB:-}" && "${BINS_JOB}" != "DRYRUN_JOB" ]]; then
  EVAL_DEP=(--dependency="afterok:${BINS_JOB}")
elif [[ ! -f "${BINS_TSV}" ]]; then
  echo "[znf282] WARN: bins missing and no bins job id; eval may skip" >&2
fi

inputs="raw=${MM} unified=${IMPUTE}"
tag="${tf}_${MODE}"
cmp_out="${DOWN_DIR}/compare_pos_neg_${tag}"
bin_out="${DOWN_DIR}/bin_coverage_${tag}"
peak_out="${DOWN_DIR}/peak_coverage_${tag}"

echo "[znf282] submitting final-mode eval"
AUROC_JOB=$(sbatch_or_echo "${EVAL_DEP[@]}" --job-name="ds_auroc_${tag}" \
  --export=ALL,BINS_TSV="${BINS_TSV}",OUT_DIR="${cmp_out}",OUT_NAME="compare_pos_neg_${tag}",INPUTS="${inputs}" \
  "${SLURM_DIR}/compare_pos_neg.sbatch")
BINCOV_JOB=$(sbatch_or_echo "${EVAL_DEP[@]}" --job-name="ds_bincov_${tag}" \
  --export=ALL,BINS_TSV="${BINS_TSV}",OUT_DIR="${bin_out}",OUT_NAME="bin_coverage_${tag}",INPUTS="${inputs}" \
  "${SLURM_DIR}/bin_sensitivity_specificity.sbatch")
PEAKCOV_JOB=$(sbatch_or_echo "${EVAL_DEP[@]}" --job-name="ds_peakcov_${tag}" \
  --export=ALL,BINS_TSV="${BINS_TSV}",PEAK_COV_OUT_DIR="${peak_out}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}" \
  "${SLURM_DIR}/peak_coverage.sbatch")

cat <<EOF

ZNF282 downstream queued.
  peaks : ${PEAKS}
  bam   : ${BAM}
  bins  : ${BINS_TSV}
  impute: ${IMPUTE}
  jobs  : bins=${BINS_JOB:-done} auroc=${AUROC_JOB} bincov=${BINCOV_JOB} peakcov=${PEAKCOV_JOB}

Monitor:
  squeue -u \$USER -n znf282_bulk_bam,ds_bins_${tf}_${MODE},ds_auroc_${tag}

When eval finishes:
  python3 downstream/collect_tf_eval_summary.py > downstream/tf_eval_summary.tsv

EOF
