#!/bin/bash
#SBATCH --job-name=hic_peakachu
#SBATCH --output=hic_peakachu-%j.out
#SBATCH --error=hic_peakachu-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#
# Peakachu chromatin loops from the balanced HEK293T Hi-C .mcool.
# Official pipeline: depth -> pretrained high-confidence model -> score_genome
# -> pool. Then annotate loops with A/B compartments + CTCF consensus peaks.
#
#   sbatch hic/slurm/06_peakachu_loops.sbatch
#   LOOP_RES=5000 sbatch hic/slurm/06_peakachu_loops.sbatch
#   LOOP_RES=10000 LOOP_THRESH=0.5 LOOP_THRESHES=0.5,0.7,0.9 \
#     sbatch hic/slurm/06_peakachu_loops.sbatch
#   PEAKACHU_MODEL=/path/to/high-confidence.150million.10kb.w6.pkl \
#     sbatch hic/slurm/06_peakachu_loops.sbatch
#
# Prereq: 02_pileup produced a balanced .mcool with ${LOOP_RES} (and 1 Mb for
# depth). 04/05 are optional but used for A/B annotation when present.

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

command -v peakachu >/dev/null 2>&1 || { echo "peakachu not on PATH in env ${CONDA_ENV} (pip install peakachu)" >&2; exit 1; }
PEAKACHU=(python "${HIC}/scripts/run_peakachu.py")

MCOOL="${MCOOL:-$(ls "${WORK}"/coolers-"${GENOME}"/*.mcool 2>/dev/null | head -1)}"
[[ -s "${MCOOL:-}" ]] || { echo "no .mcool under ${WORK}/coolers-${GENOME}/ (run 02)" >&2; exit 1; }

RES="${LOOP_RES}"
THRESH="${LOOP_THRESH}"
THRESHES="${LOOP_THRESHES:-${THRESH}}"
WEIGHT="${LOOP_WEIGHT}"
COOL="${MCOOL}::/resolutions/${RES}"
DEPTH_COOL="${MCOOL}::/resolutions/${LOOP_DEPTH_RES}"

# Peakachu Hi-C pretrained table is 5 / 10 / 25 kb only. This mcool finest is 5 kb.
if [[ "${RES}" -lt 5000 ]]; then
  echo "LOOP_RES=${RES}: Peakachu has no official Hi-C model below 5 kb, and this mcool finest is 5 kb." >&2
  echo "available: $(cooler ls "${MCOOL}" | sed 's#.*/##' | tr '\n' ' ')" >&2
  exit 2
fi

cooler info "${COOL}" >/dev/null 2>&1 || {
  echo "missing ${RES} bp in ${MCOOL}; available: $(cooler ls "${MCOOL}" | sed 's#.*/##' | tr '\n' ' ')" >&2
  exit 1
}

OUT="${WORK}/loops"
MODEL_DIR="${OUT}/models"
mkdir -p "${OUT}" "${MODEL_DIR}"
cd "${OUT}"

# 25 kb models use w5; 5/10 kb models use w6 (Peakachu v2 pretrained table).
if [[ "${RES}" -eq 25000 ]]; then WIN="w5"; else WIN="w6"; fi

echo "[$(date)] Peakachu loops  mcool=${MCOOL}  res=${RES}  thresh=${THRESH}  threshes=${THRESHES}  weight=${WEIGHT}"

if [[ -n "${PEAKACHU_MODEL}" ]]; then
  MODEL="${PEAKACHU_MODEL}"
  [[ -s "${MODEL}" ]] || { echo "PEAKACHU_MODEL not found: ${MODEL}" >&2; exit 1; }
  echo "[$(date)] using provided model ${MODEL}"
else
  cooler info "${DEPTH_COOL}" >/dev/null 2>&1 || {
    echo "missing ${LOOP_DEPTH_RES} bp in ${MCOOL} (needed for peakachu depth)" >&2
    exit 1
  }
  echo "[$(date)] peakachu depth on ${DEPTH_COOL}"
  "${PEAKACHU[@]}" depth -p "${DEPTH_COOL}" | tee "${OUT}/peakachu_depth.txt"
  # "suggested model: 150 million" / "1.8 billion" / "2 billion"
  SUGGESTED=$(awk -F': *' 'tolower($0) ~ /suggested model/ {print $2; exit}' "${OUT}/peakachu_depth.txt")
  SUGGESTED="${SUGGESTED:-}"
  DEPTH_TAG=$(python - "${SUGGESTED}" <<'PY'
import re, sys
s = sys.argv[1].strip().lower().replace(" ", "")
if not s:
    sys.exit("could not parse peakachu depth suggestion")
# 150million / 1.8billion / 2billion / 1billion
if not re.fullmatch(r"\d+(\.\d+)?(million|billion)", s):
    sys.exit(f"unrecognized depth tag: {sys.argv[1]!r}")
print(s)
PY
)
  MODEL="${MODEL_DIR}/high-confidence.${DEPTH_TAG}.${RES}kb.${WIN}.pkl"
  URL="http://3dgenome.fsm.northwestern.edu/peakachu/high-confidence.${DEPTH_TAG}.${RES}kb.${WIN}.pkl"
  if [[ ! -s "${MODEL}" ]]; then
    echo "[$(date)] download ${URL}"
    curl -fL --retry 5 --retry-delay 10 \
      -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
      -o "${MODEL}.tmp" "${URL}"
    if file "${MODEL}.tmp" | grep -q "XZ compressed"; then
      xz -dc "${MODEL}.tmp" > "${MODEL}"
      rm -f "${MODEL}.tmp"
    else
      mv "${MODEL}.tmp" "${MODEL}"
    fi
  else
    echo "[$(date)] cached model ${MODEL}"
  fi
fi

SCORES="${OUT}/peakachu_${RES}.scores.bedpe"

if [[ -s "${SCORES}" && "${FORCE_SCORE}" != "1" ]]; then
  echo "[$(date)] reuse existing scores ${SCORES} (FORCE_SCORE=1 to re-score)"
else
  echo "[$(date)] score_genome  model=$(basename "${MODEL}")  min_prob=${LOOP_MIN_PROB}"
  "${PEAKACHU[@]}" score_genome \
    -r "${RES}" \
    --clr-weight-name "${WEIGHT}" \
    --minimum-prob "${LOOP_MIN_PROB}" \
    -p "${COOL}" \
    -O "${SCORES}" \
    -m "${MODEL}"
fi

AB="${WORK}/compartments/compartments_${COMPARTMENT_RES}.AB.bed"
ANN_ARGS=()
if [[ -s "${AB}" ]]; then
  ANN_ARGS+=(--ab-bed "${AB}")
fi
if [[ -s "${CTCF_PEAKS}" ]]; then
  ANN_ARGS+=(--ctcf "${CTCF_PEAKS}")
fi

IFS=',' read -r -a THRESH_ARR <<< "${THRESHES}"
# always include the primary threshold
if [[ ! " ${THRESH_ARR[*]} " =~ " ${THRESH} " ]]; then
  THRESH_ARR=("${THRESH}" "${THRESH_ARR[@]}")
fi

PRIMARY_LOOPS=""
for T in "${THRESH_ARR[@]}"; do
  T="${T// /}"
  [[ -n "${T}" ]] || continue
  LOOPS="${OUT}/peakachu_${RES}.loops.${T}.bedpe"
  echo "[$(date)] pool  thresh=${T}"
  "${PEAKACHU[@]}" pool \
    -r "${RES}" \
    -i "${SCORES}" \
    -o "${LOOPS}" \
    -t "${T}"
  echo "[$(date)] annotate loops  thresh=${T}"
  python "${HIC}/scripts/analyze_loops.py" \
    --loops "${LOOPS}" \
    --out-prefix "${OUT}/peakachu_${RES}.loops.${T}" \
    "${ANN_ARGS[@]}"
  echo "  loops:      ${LOOPS}  (n=$(wc -l < "${LOOPS}"))"
  echo "  summary:    ${OUT}/peakachu_${RES}.loops.${T}.summary.tsv"
  if [[ "${T}" == "${THRESH}" ]]; then
    PRIMARY_LOOPS="${LOOPS}"
  fi
done

echo "[$(date)] done  scores=${SCORES}  primary=${PRIMARY_LOOPS:-${OUT}/peakachu_${RES}.loops.${THRESH}.bedpe}"
