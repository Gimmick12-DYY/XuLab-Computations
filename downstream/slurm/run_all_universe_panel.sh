#!/bin/bash
# -----------------------------------------------------------------------------
# run_all_universe_panel.sh
#
# Slide-panel eval with the full positive/negative universes:
#   positives = all_bound      (every TF-bound cCRE bin, post-blacklist)
#   negatives = all_candidates (every zero-bulk-read non-TF-bound bin)
#
# Outputs under mode tag `all_universe`:
#   downstream/bins_<tf>_all_universe/
#   downstream/compare_pos_neg_<tf>_all_universe/
#   downstream/bin_coverage_<tf>_all_universe/
#   downstream/peak_coverage_<tf>_all_universe/
#
# Usage:
#   bash downstream/slurm/run_all_universe_panel.sh
#   bash downstream/slurm/run_all_universe_panel.sh MAZ ZNF282
#   SKIP_EVAL=1 bash downstream/slurm/run_all_universe_panel.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DOWN_DIR="${ROOT_DIR}/downstream"
SLURM_DIR="${DOWN_DIR}/slurm"
SBATCH_BINS="${SLURM_DIR}/prepare_pos_neg_bins.sbatch"
DEFAULT_TFS=(CTCF MAZ ZIC2 ZBTB7A ZNF777 NFYA ZNF282)
MODE=all_universe
SKIP_EVAL="${SKIP_EVAL:-0}"
# Shared cCRE/blacklist cache (avoid re-download).
CACHE_DIR="${CACHE_DIR:-${DOWN_DIR}/bins_ctcf_all_bound/cache}"

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

  echo "[submit] ${TF} POS_MODE=all_bound NEGATIVE_MODE=all_candidates -> ${OUT}"
  bin_job="$(sbatch --parsable --time=06:00:00 --mem=64G \
    --job-name="ds_bins_${tf}_${MODE}" --export=ALL,\
MM_DIR="${MM}",\
OUT_DIR="${OUT}",\
CACHE_DIR="${CACHE_DIR}",\
TF_PEAKS_BED="${PEAKS}",\
BULK_BAM="${BULK}",\
POS_MODE=all_bound,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=all_candidates,\
MAX_NEG=0,\
SEED=2026 \
    "${SBATCH_BINS}")"
  echo "[submit] bins job=${bin_job}"

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    continue
  fi

  tag="${tf}_${MODE}"
  bins_tsv="${OUT}/pos_neg_bins.tsv"
  inputs="raw=${MM} unified=${IMPUTE}"
  # Prefer all_bound MACS peaks; ZNF282 falls back to final / peak_cutoff_q05.
  peak_base=""
  for cand in \
    "${DOWN_DIR}/peak_coverage_${tf}_all_bound" \
    "${DOWN_DIR}/peak_coverage_${tf}_final" \
    "${DOWN_DIR}/peak_coverage_${tf}_peak_cutoff_q05"
  do
    if [[ -d "${cand}/raw" && -d "${cand}/unified" ]]; then
      peak_base="${cand}"
      break
    fi
  done
  cmp_out="${DOWN_DIR}/compare_pos_neg_${tag}"
  bin_out="${DOWN_DIR}/bin_coverage_${tag}"
  peak_out="${DOWN_DIR}/peak_coverage_${tag}"

  # Export var NAMES (not VAR=value) so INPUTS can contain spaces
  # (raw=... unified=...). Embedding INPUTS=... in --export truncates at space.
  export BINS_TSV="${bins_tsv}"
  export INPUTS="${inputs}"

  # Larger pos/neg sets (esp. CTCF) need more mem/time than the defaults.
  export OUT_DIR="${cmp_out}"
  export OUT_NAME="compare_pos_neg_${tag}"
  auroc_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
    --mem=128G --time=06:00:00 \
    --job-name="ds_auroc_${tag}" \
    --export=ALL,BINS_TSV,OUT_DIR,OUT_NAME,INPUTS \
    "${SLURM_DIR}/compare_pos_neg.sbatch")"
  echo "[submit] auroc job=${auroc_job} (afterok:${bin_job})"

  export OUT_DIR="${bin_out}"
  export OUT_NAME="bin_coverage_${tag}"
  bincov_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
    --mem=128G --time=04:00:00 \
    --job-name="ds_bincov_${tag}" \
    --export=ALL,BINS_TSV,OUT_DIR,OUT_NAME,INPUTS \
    "${SLURM_DIR}/bin_sensitivity_specificity.sbatch")"
  echo "[submit] bincov job=${bincov_job} (afterok:${bin_job})"

  export PEAK_COV_OUT_DIR="${peak_out}"
  export OUT_NAME="peak_coverage_${tag}"
  if [[ -z "${peak_base}" ]]; then
    echo "[warn] ${TF}: no reusable peaks dir; peak coverage will call MACS de novo" >&2
    unset REUSE_PEAKS_FROM || true
    peak_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
      --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV,PEAK_COV_OUT_DIR,OUT_NAME,INPUTS \
      "${SLURM_DIR}/peak_coverage.sbatch")"
  else
    echo "[submit] ${TF}: REUSE_PEAKS_FROM=${peak_base}"
    export REUSE_PEAKS_FROM="${peak_base}"
    peak_job="$(sbatch --parsable --dependency="afterok:${bin_job}" \
      --mem=64G --time=04:00:00 \
      --job-name="ds_peakcov_${tag}" \
      --export=ALL,BINS_TSV,PEAK_COV_OUT_DIR,OUT_NAME,INPUTS,REUSE_PEAKS_FROM \
      "${SLURM_DIR}/peak_coverage.sbatch")"
  fi
  echo "[submit] peakcov job=${peak_job} (afterok:${bin_job})"
done

cat <<EOF

Queued all_universe (all_bound positives + all_candidates negatives).

After jobs finish, build the slide table:
  python3 downstream/build_slide_all_universe.py

Monitor:
  squeue -u \$USER -n ds_bins_,ds_auroc_,ds_bincov_,ds_peakcov_

EOF
