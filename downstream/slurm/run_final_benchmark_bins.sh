#!/bin/bash
# -----------------------------------------------------------------------------
# run_final_benchmark_bins.sh
#
# Final positive-region definition (one set per TF):
#   CTCF, MAZ, ZIC2  -> top ~12k positives
#     CTCF: SCREEN top-10k CTCF cCREs (Bin_posVSneg.ipynb; ~12,805 bins)
#     MAZ / ZIC2: top 12,000 TF-bound cCRE bins by bulk peak signal
#   remaining TFs    -> all available TF-bound cCRE bins (all_bound)
#   negatives        -> zero-read bedcov pool, randomly matched to #positives
#
# Outputs: downstream/bins_<tf>_final/pos_neg_bins.tsv
#
# Usage:
#   bash downstream/slurm/run_final_benchmark_bins.sh
#   DRY_RUN=1 bash downstream/slurm/run_final_benchmark_bins.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
SBATCH="${SBATCH:-${DOWN_DIR}/slurm/prepare_pos_neg_bins.sbatch}"
SEED="${SEED:-2026}"
N_TOP12K="${N_TOP12K:-12805}"

TOP12K_TFS=(CTCF MAZ ZIC2)
ALL_BOUND_TFS=(ZBTB7A ZNF777 NFYA ZNF282)

sbatch_or_echo() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] sbatch $*"
  else
    sbatch "$@"
  fi
}

resolve_paths() {
  local TF="$1"
  local tf="${TF,,}"
  local line="HEK293"
  [[ "${TF}" == "NFYA" ]] && line="K562"

  MM_DIR="${ROOT_DIR}/unified/work/${tf}/mm"
  PEAKS_BED="${DATA_DIR}/${TF}_${line}_peaks.bed"
  if [[ "${TF}" == "CTCF" && -f "${DATA_DIR}/ENCFF139DCW.bam" ]]; then
    BULK_BAM="${DATA_DIR}/ENCFF139DCW.bam"
  else
    BULK_BAM="${DATA_DIR}/${TF}_${line}.bam"
  fi
  OUT_DIR="${DOWN_DIR}/bins_${tf}_final"
}

# --- CTCF: reuse original SCREEN top-10k bins (exact Bin_posVSneg definition) ---
resolve_paths CTCF
LEGACY_CTCF_BINS="${DOWN_DIR}/bins"
if [[ ! -f "${LEGACY_CTCF_BINS}/pos_neg_bins.tsv" ]]; then
  echo "ERROR: missing legacy CTCF bins at ${LEGACY_CTCF_BINS}/pos_neg_bins.tsv" >&2
  exit 1
fi
echo "[final] CTCF: copy SCREEN top-10k bins (${LEGACY_CTCF_BINS} -> ${OUT_DIR})"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
  cp -f "${LEGACY_CTCF_BINS}/pos_neg_bins.tsv" "${OUT_DIR}/pos_neg_bins.tsv"
  if [[ -f "${LEGACY_CTCF_BINS}/pos_neg_bins.meta.json" ]]; then
    cp -f "${LEGACY_CTCF_BINS}/pos_neg_bins.meta.json" "${OUT_DIR}/pos_neg_bins.meta.json"
  fi
  # Annotate that this is the final-mode copy of the notebook definition.
  python3 - <<'PY' "${OUT_DIR}/pos_neg_bins.meta.json" || true
import json, sys, re
path = sys.argv[1]
try:
    text = open(path).read()
    text = text.replace(": TRUE", ": true").replace(": FALSE", ": false").replace(": NA", ": null")
    d = json.loads(text)
except Exception:
    raise SystemExit(0)
d["final_mode"] = "top12k_SCREEN_cCRE"
d["final_note"] = "Copied from downstream/bins (Bin_posVSneg top-10k CTCF cCREs)"
open(path, "w").write(json.dumps(d, indent=2) + "\n")
PY
fi

# --- MAZ / ZIC2: top 12k by peak signal ---
for TF in MAZ ZIC2; do
  resolve_paths "${TF}"
  echo "[final] ${TF}: POS_MODE=fixed_n N_POS=${N_TOP12K} -> ${OUT_DIR}"
  for f in "${PEAKS_BED}" "${BULK_BAM}" "${MM_DIR}/regions.tsv.gz"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
  done
  sbatch_or_echo --job-name="ds_bins_${TF,,}_final" \
    --export=ALL,\
MM_DIR="${MM_DIR}",\
OUT_DIR="${OUT_DIR}",\
TF_PEAKS_BED="${PEAKS_BED}",\
BULK_BAM="${BULK_BAM}",\
POS_MODE=fixed_n,\
N_POS="${N_TOP12K}",\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED="${SEED}" \
    "${SBATCH}"
done

# --- Remaining TFs: all available positives (reuse existing all_bound) ---
for TF in "${ALL_BOUND_TFS[@]}"; do
  resolve_paths "${TF}"
  src="${DOWN_DIR}/bins_${TF,,}_all_bound"
  if [[ ! -f "${src}/pos_neg_bins.tsv" ]]; then
    echo "[final] ${TF}: all_bound bins missing; submitting prep -> ${OUT_DIR}"
    for f in "${PEAKS_BED}" "${BULK_BAM}" "${MM_DIR}/regions.tsv.gz"; do
      [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
    done
    sbatch_or_echo --job-name="ds_bins_${TF,,}_final" \
      --export=ALL,\
MM_DIR="${MM_DIR}",\
OUT_DIR="${OUT_DIR}",\
TF_PEAKS_BED="${PEAKS_BED}",\
BULK_BAM="${BULK_BAM}",\
POS_MODE=all_bound,\
N_POS=0,\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED="${SEED}" \
      "${SBATCH}"
  else
    echo "[final] ${TF}: symlink all_bound -> ${OUT_DIR}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      rm -rf "${OUT_DIR}"
      ln -sfn "${src}" "${OUT_DIR}"
    fi
  fi
done

cat <<EOF

Final bin definition queued / prepared:
  CTCF, MAZ, ZIC2 : top ~12k positives (CTCF=SCREEN 10k cCREs; MAZ/ZIC2=top ${N_TOP12K} by signal)
  ZBTB7A, ZNF777, NFYA, ZNF282 : all_bound
  negatives : matched to #positives (zero-read bedcov pool)

ZNF282 bulk (GEO GSM2466515, not ENCODE):
  bash data_prep/fetch_znf282_bulk.sh --align   # peaks + BAM

Monitor MAZ/ZIC2 prep:
  squeue -u \$USER -n ds_bins_maz_final,ds_bins_zic2_final

Then run:
  bash downstream/slurm/run_final_downstream_eval.sh

EOF
