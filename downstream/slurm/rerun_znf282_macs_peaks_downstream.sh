#!/bin/bash
# -----------------------------------------------------------------------------
# rerun_znf282_macs_peaks_downstream.sh
#
# 1) Call hg38 peaks from data/ZNF282_HEK293.bam (MACS3) — replaces liftOver
# 2) Rebuild final (all_bound) + peak_cutoff_q05 bin sets
# 3) Rerun AUROC / bin / peak coverage for those modes
# 4) Rebuild slide table (top12k + q05)
#
# Usage (repo root):
#   bash downstream/slurm/rerun_znf282_macs_peaks_downstream.sh
#   DRY_RUN=1 bash downstream/slurm/rerun_znf282_macs_peaks_downstream.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"
DOWN_DIR="${ROOT_DIR}/downstream"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
SLURM_DIR="${DOWN_DIR}/slurm"
TF=ZNF282
tf=znf282

PEAKS="${DATA_DIR}/${TF}_HEK293_peaks.bed"
BAM="${DATA_DIR}/${TF}_HEK293.bam"
MM="${ROOT_DIR}/unified/work/${tf}/mm"
IMPUTE="${ROOT_DIR}/unified/work/${tf}/impute"

sbatch_or_echo() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] sbatch $*" >&2
    echo "DRYRUN_JOB"
  else
    sbatch --parsable "$@"
  fi
}

for f in "${BAM}" "${BAM}.bai" "${MM}/regions.tsv.gz"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
[[ -d "${IMPUTE}" ]] || { echo "ERROR: missing impute ${IMPUTE}" >&2; exit 1; }

echo "[znf282] submit MACS3 peak call from BAM"
PEAK_JOB=$(sbatch_or_echo --job-name="znf282_macs_peaks" \
  "${ROOT_DIR}/data_prep/slurm/call_znf282_peaks.sbatch")
echo "[znf282] peak job=${PEAK_JOB}"
PEAK_DEP=(--dependency="afterok:${PEAK_JOB}")
[[ "${PEAK_JOB}" == "DRYRUN_JOB" ]] && PEAK_DEP=()

# Clear stale bin dirs so they are rebuilt against the new peaks.
for mode in final peak_cutoff_q05; do
  d="${DOWN_DIR}/bins_${tf}_${mode}"
  if [[ -e "${d}" ]]; then
    echo "[znf282] remove stale bins ${d}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      rm -rf "${d}"
    fi
  fi
done

echo "[znf282] submit final all_bound bins"
FINAL_BINS=$(sbatch_or_echo "${PEAK_DEP[@]}" --time=04:00:00 \
  --job-name="ds_bins_${tf}_final" \
  --export=ALL,\
MM_DIR="${MM}",\
OUT_DIR="${DOWN_DIR}/bins_${tf}_final",\
TF_PEAKS_BED="${PEAKS}",\
BULK_BAM="${BAM}",\
POS_MODE=all_bound,\
N_POS=0,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED=2026 \
  "${SLURM_DIR}/prepare_pos_neg_bins.sbatch")
echo "[znf282] final bins job=${FINAL_BINS}"

echo "[znf282] submit peak_cutoff_q05 bins"
Q05_BINS=$(sbatch_or_echo "${PEAK_DEP[@]}" --time=04:00:00 \
  --job-name="ds_bins_${tf}_q05" \
  --export=ALL,\
MM_DIR="${MM}",\
OUT_DIR="${DOWN_DIR}/bins_${tf}_peak_cutoff_q05",\
TF_PEAKS_BED="${PEAKS}",\
BULK_BAM="${BAM}",\
POS_MODE=peak_cutoff,\
PEAK_Q_MAX=0.05,\
PEAK_P_MAX=1e-5,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED=2026 \
  "${SLURM_DIR}/prepare_pos_neg_bins.sbatch")
echo "[znf282] q05 bins job=${Q05_BINS}"

submit_eval() {
  local mode="$1" bins_job="$2"
  local tag="${tf}_${mode}"
  local bins="${DOWN_DIR}/bins_${tf}_${mode}/pos_neg_bins.tsv"
  local inputs="raw=${MM} unified=${IMPUTE}"
  local dep=(--dependency="afterok:${bins_job}")
  [[ "${bins_job}" == "DRYRUN_JOB" ]] && dep=()

  local auroc
  auroc=$(sbatch_or_echo "${dep[@]}" --job-name="ds_auroc_${tag}" \
    --export=ALL,BINS_TSV="${bins}",OUT_DIR="${DOWN_DIR}/compare_pos_neg_${tag}",OUT_NAME="compare_pos_neg_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/compare_pos_neg.sbatch")
  echo "[znf282] auroc ${tag} job=${auroc}"

  local bincov
  bincov=$(sbatch_or_echo "${dep[@]}" --job-name="ds_bincov_${tag}" \
    --export=ALL,BINS_TSV="${bins}",OUT_DIR="${DOWN_DIR}/bin_coverage_${tag}",OUT_NAME="bin_coverage_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/bin_sensitivity_specificity.sbatch")
  echo "[znf282] bincov ${tag} job=${bincov}"

  local peakcov
  peakcov=$(sbatch_or_echo "${dep[@]}" --job-name="ds_peakcov_${tag}" \
    --export=ALL,BINS_TSV="${bins}",PEAK_COV_OUT_DIR="${DOWN_DIR}/peak_coverage_${tag}",OUT_NAME="peak_coverage_${tag}",INPUTS="${inputs}" \
    "${SLURM_DIR}/peak_coverage.sbatch")
  echo "[znf282] peakcov ${tag} job=${peakcov}"
}

submit_eval final "${FINAL_BINS}"
submit_eval peak_cutoff_q05 "${Q05_BINS}"

# Slide table after q05 bins complete
SLIDE_DEP=(--dependency="afterok:${Q05_BINS}")
[[ "${Q05_BINS}" == "DRYRUN_JOB" ]] && SLIDE_DEP=()
SLIDE_JOB=$(sbatch_or_echo "${SLIDE_DEP[@]}" \
  --job-name="ds_slide_top12k_q05" \
  --mem=4G --time=00:20:00 \
  --wrap="cd '${ROOT_DIR}' && python3 downstream/build_slide_top12k_q05.py")
echo "[znf282] slide table job=${SLIDE_JOB}"

cat <<EOF

Queued ZNF282 BAM→MACS peaks + downstream rebuild.
  peaks : ${PEAK_JOB}
  bins  : final=${FINAL_BINS}  q05=${Q05_BINS}
  slide : ${SLIDE_JOB}

Monitor:
  squeue -u \$USER -n znf282_macs_peaks,ds_bins_${tf}_final,ds_bins_${tf}_q05

LiftOver peaks will be archived to:
  ${DATA_DIR}/ZNF282_HEK293_peaks.liftOver.bed
EOF
