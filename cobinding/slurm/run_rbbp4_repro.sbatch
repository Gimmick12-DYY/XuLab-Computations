#!/bin/bash
#SBATCH --job-name=rbbp4_repro
#SBATCH --output=rbbp4_repro-%j.out
#SBATCH --error=rbbp4_repro-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL
#
# Sidecar RBBP4 Cicero rebuild. Does NOT touch cobinding/results/RBBP4 or
# cobinding/work/RBBP4.peak_edges.tsv (the gold-standard fitConns run).
#
# Peak universe = unique intervals in data/TF.RBBP4.fitConns.res.sel (20,041).
# Signal = raw HiTAG unified/work/rbbp4/mm. Then real run_cicero(window=1e6).
#
#   sbatch cobinding/slurm/run_rbbp4_repro.sbatch
#   REBUILD_CICERO=1 sbatch cobinding/slurm/run_rbbp4_repro.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${XULAB}/Cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
TAG="${TAG:-RBBP4_repro}"
WORK="${COB_WORK:-${COB}/work/${TAG}}"
RESULTS="${COB_RESULTS:-${COB}/results/${TAG}}"
MM_RAW="${UNIFIED_MM:-${XULAB}/unified/work/rbbp4/mm}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
CFG="${CFG:-${COB}/configs/cicero_1mb.yaml}"
FITCONNS="${FITCONNS:-${DATA_DIR}/TF.RBBP4.fitConns.res.sel}"
GOLD_EDGES="${GOLD_EDGES:-${COB}/work/RBBP4.peak_edges.tsv}"
GOLD_SUMMARY="${GOLD_SUMMARY:-${COB}/results/RBBP4/summary.txt}"

TF=RBBP4
QVAL="${QVAL:-0.05}"
COACCESS_MIN="${COACCESS_MIN:-0.05}"
MAX_DIST="${MAX_DIST:-1000000}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"
SHARE="${SHARE:-2}"
NET_MIN_COMP="${NET_MIN_COMP:-3}"
NET_BIG="${NET_BIG:-10}"
REBUILD_CICERO="${REBUILD_CICERO:-0}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PY_ENV}"

mkdir -p "${WORK}" "${RESULTS}" "${COB}/slurm"
echo "[$(date)] sidecar ${TAG}  work=${WORK}  results=${RESULTS}"
echo "[$(date)] raw mm=${MM_RAW}  fitConns=${FITCONNS}"

PEAKS="${WORK}/fitconns_peaks.bed"
python "${COB}/scripts/extract_fitconns_peaks.py" \
  --fitconns "${FITCONNS}" --out-bed "${PEAKS}"

CWD="${WORK}/cicero"
CONNS="${CWD}/cicero/coaccess.tsv.gz"
EDGES="${WORK}/${TF}.peak_edges.tsv"

if [[ "${REBUILD_CICERO}" != "1" && -s "${CONNS}" ]]; then
  echo "[$(date)] reuse ${CONNS}"
else
  echo "[$(date)] export peak x cell mm from raw HiTAG onto fitConns peaks"
  python "${COB}/scripts/export_tf_cicero_mm.py" \
    --tf "${TF}" --peaks "${PEAKS}" --mm-dir "${MM_RAW}" \
    --out-mm "${CWD}/mm" --max-peaks 0

  echo "[$(date)] Cicero CDS + run_cicero window=${MAX_DIST}"
  conda activate "${CICERO_ENV}"
  module unload r 2>/dev/null || true
  cd "${CICERO_ROOT}"
  Rscript scripts/check_env.R
  Rscript scripts/02_build_cds.R --config "${CFG}" --work-dir "${CWD}"
  Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}"
  cd "${XULAB}"
  conda activate "${PY_ENV}"
fi

echo "[$(date)] Cicero conns -> cobinding edges (dedup unordered pairs)"
conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf "${TF}" --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist "${MAX_DIST}" \
  --coaccess-min "${COACCESS_MIN}"

echo "[$(date)] cluster_peaks (same knobs as gold RBBP4)"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${EDGES}" --out-dir "${RESULTS}" \
  --qval 1 --coaccess-min 0 --min-support 1 \
  --min-clique "${MIN_CLIQUE}" --share "${SHARE}" --plot \
  --net-min-comp "${NET_MIN_COMP}" --net-big "${NET_BIG}" \
  --max-dist "${MAX_DIST}"

PDC="${WORK}/${TF}.generated.pdc"
if [[ -s "${PDC}" && -s "${CHROMHMM}" ]]; then
  echo "[$(date)] distal_chromhmm"
  python "${COB}/scripts/distal_chromhmm_composition.py" \
    --conns "${PDC}" --chromhmm "${CHROMHMM}" \
    --out-dir "${RESULTS}" --tf "${TF}" --qval 1
fi

echo "[$(date)] compare against gold fitConns RBBP4"
python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" \
  --repro-edges "${EDGES}" \
  --gold-summary "${GOLD_SUMMARY}" \
  --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" \
  --qval "${QVAL}" --coaccess-min "${COACCESS_MIN}"

echo "[$(date)] done -> ${RESULTS}"
cat "${RESULTS}/compare_to_gold.txt"
