#!/bin/bash
#SBATCH --job-name=rbbp4_labpmat
#SBATCH --output=rbbp4_labpmat-%j.out
#SBATCH --error=rbbp4_labpmat-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# Sidecar Cicero on the lab RBBP4-specific pmat (53,825 x 181,396).
# Does NOT touch cobinding/results/RBBP4, cobinding/work/RBBP4.peak_edges.tsv,
# or anything under /vast/som/xujie_lab/.
#
#   sbatch cobinding/slurm/run_rbbp4_labpmat.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
TAG="${TAG:-RBBP4_labpmat}"
WORK="${COB_WORK:-${COB}/work/${TAG}}"
RESULTS="${COB_RESULTS:-${COB}/results/${TAG}}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
CFG="${CFG:-${COB}/configs/cicero_1mb_labpmat.yaml}"
GOLD_EDGES="${GOLD_EDGES:-${COB}/work/RBBP4.peak_edges.tsv}"
GOLD_SUMMARY="${GOLD_SUMMARY:-${COB}/results/RBBP4/summary.txt}"
LAB_PMAT_SRC="${LAB_PMAT_SRC:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/pmat/seurat/RBBP4.pmat.mtx.rds}"
LAB_CONNS_SRC="${LAB_CONNS_SRC:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds}"
PMAT="${PMAT:-${DATA_DIR}/RBBP4.pmat.mtx.rds}"

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

mkdir -p "${WORK}" "${RESULTS}"
echo "[$(date)] sidecar ${TAG}  work=${WORK}  results=${RESULTS}"

# Copy lab pmat into OUR space (read-only source).
if [[ ! -s "${PMAT}" ]]; then
  [[ -s "${LAB_PMAT_SRC}" ]] || { echo "lab RBBP4 pmat not found: ${LAB_PMAT_SRC}" >&2; exit 1; }
  echo "[$(date)] copy lab RBBP4.pmat -> ${PMAT} (lab share not modified)"
  cp -n "${LAB_PMAT_SRC}" "${PMAT}"
fi

CWD="${WORK}/cicero"
CONNS="${CWD}/cicero/coaccess.tsv.gz"
EDGES="${WORK}/${TF}.peak_edges.tsv"

if [[ "${REBUILD_CICERO}" != "1" && -s "${CONNS}" ]]; then
  echo "[$(date)] reuse ${CONNS}"
else
  echo "[$(date)] export lab RBBP4 pmat -> Cicero mm (all 181k cells, min-cells-per-peak=1)"
  conda activate "${CICERO_ENV}"
  Rscript "${COB}/scripts/export_pmat_tf_mm.R" \
    --pmat "${PMAT}" --out-mm "${CWD}/mm" \
    --min-cells-per-peak 1 --min-peaks-per-cell 0

  echo "[$(date)] Cicero CDS + run_cicero window=${MAX_DIST}"
  module unload r 2>/dev/null || true
  cd "${CICERO_ROOT}"
  Rscript scripts/check_env.R
  Rscript scripts/02_build_cds.R --config "${CFG}" --work-dir "${CWD}"
  Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}"
  cd "${XULAB}"
fi

echo "[$(date)] compare raw conns to lab RBBP4.conns.rds (read-only)"
conda activate "${CICERO_ENV}"
if [[ -s "${LAB_CONNS_SRC}" && -s "${CONNS}" ]]; then
  Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
    --lab-conns "${LAB_CONNS_SRC}" \
    --ours "${CONNS}" \
    --out "${RESULTS}/compare_to_lab_conns.txt"
fi

echo "[$(date)] Cicero conns -> cobinding edges"
conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf "${TF}" --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist "${MAX_DIST}" \
  --coaccess-min "${COACCESS_MIN}"

echo "[$(date)] cluster_peaks (sidecar ${TAG})"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${EDGES}" --out-dir "${RESULTS}" \
  --qval 1 --coaccess-min 0 --min-support 1 \
  --min-clique "${MIN_CLIQUE}" --share "${SHARE}" --plot \
  --net-min-comp "${NET_MIN_COMP}" --net-big "${NET_BIG}" \
  --max-dist "${MAX_DIST}"

echo "[$(date)] compare against gold fitConns RBBP4"
python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" \
  --repro-edges "${EDGES}" \
  --gold-summary "${GOLD_SUMMARY}" \
  --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" \
  --qval "${QVAL}" --coaccess-min "${COACCESS_MIN}"

echo "[$(date)] done -> ${RESULTS}"
[[ -s "${RESULTS}/compare_to_lab_conns.txt" ]] && cat "${RESULTS}/compare_to_lab_conns.txt"
echo "----"
cat "${RESULTS}/compare_to_gold.txt"
