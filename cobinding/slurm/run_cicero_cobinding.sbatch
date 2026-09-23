#!/bin/bash
#SBATCH --job-name=cob_cicero
#SBATCH --output=cob_cicero-%A_%a.out
#SBATCH --error=cob_cicero-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL
#
# Per-TF Cicero (1 Mb window) then the same cobinding plots as RBBP4.
# The cobinding parser never ran Cicero — it only read TF.*.fitConns files,
# which exist solely for RBBP4. This job is the missing step.
#
#   N=$(python cobinding/scripts/list_cobinding_tfs.py | wc -l)
#   sbatch --array=0-$((N-1))%8 cobinding/slurm/run_cicero_cobinding.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${XULAB}/Cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
WORK="${COB_WORK:-${COB}/work}"
RESULTS="${COB_RESULTS:-${COB}/results}"
UNIFIED="${UNIFIED:-${XULAB}/unified/work}"
PEAKS_IMP="${PEAKS_IMP:-${XULAB}/tf_complex/work/peaks_imputed}"
PEAKS_RAW="${PEAKS_RAW:-${XULAB}/tf_complex/work/peaks}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
CFG="${CFG:-${COB}/configs/cicero_1mb.yaml}"

QVAL="${QVAL:-0.05}"
COACCESS_MIN="${COACCESS_MIN:-0.05}"
MAX_DIST="${MAX_DIST:-1000000}"
MAX_PEAKS="${MAX_PEAKS:-20000}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"
SHARE="${SHARE:-2}"
NET_MIN_COMP="${NET_MIN_COMP:-3}"
NET_BIG="${NET_BIG:-10}"
REBUILD_CICERO="${REBUILD_CICERO:-0}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${TF:-}" ]]; then
  TF_ARR=("${TF}")
  IDX=0
else
  mapfile -t TF_ARR < <("${PY_ENV}/bin/python" "${COB}/scripts/list_cobinding_tfs.py")
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
fi
if (( IDX < 0 || IDX >= ${#TF_ARR[@]} )); then
  echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2
  exit 0
fi
TF="${TF_ARR[$IDX]}"
tf="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
echo "[$(date)] TF=${TF} idx=${IDX} window=${MAX_DIST}"

mkdir -p "${WORK}" "${RESULTS}/${TF}"
EDGES="${WORK}/${TF}.peak_edges.tsv"
CWD="${WORK}/${TF}/cicero"
CONNS="${CWD}/cicero/coaccess.tsv.gz"

HAVE_FIT=0
for cand in "${DATA_DIR}/TF.${TF}.fitConns.res.sel" "${DATA_DIR}/TF.${tf}.fitConns.res.sel" \
            "${DATA_DIR}/TF.${TF}.fitConns.res.pdc" "${DATA_DIR}/TF.${tf}.fitConns.res.pdc"; do
  [[ -s "${cand}" ]] && HAVE_FIT=1 && break
done

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${HAVE_FIT}" == "1" ]]; then
  echo "[$(date)] ${TF}: using lab Cicero fitConns (not re-running Cicero)"
  conda activate "${PY_ENV}"
  GLOB='TF.{tf}.fitConns.res.*'
  python "${COB}/scripts/build_peak_graph.py" \
    --data-dir "${DATA_DIR}" --tf "${TF}" --glob "${GLOB}" \
    --out-dir "${WORK}" --q-keep 1.0 --max-dist "${MAX_DIST}"
  PDC=""
  for cand in "${DATA_DIR}/TF.${TF}.fitConns.res.pdc" "${DATA_DIR}/TF.${tf}.fitConns.res.pdc"; do
    [[ -s "${cand}" ]] && PDC="${cand}" && break
  done
  CLUSTER_QVAL="${QVAL}"
  CLUSTER_COMIN=0.0
else
  if [[ "${REBUILD_CICERO}" != "1" && -s "${CONNS}" ]]; then
    echo "[$(date)] reuse ${CONNS}"
  else
    PEAKS=""
    if [[ -s "${PEAKS_RAW}/${tf}.bed" ]]; then
      PEAKS="${PEAKS_RAW}/${tf}.bed"
    elif [[ -s "${PEAKS_IMP}/${tf}.bed" ]]; then
      PEAKS="${PEAKS_IMP}/${tf}.bed"
    else
      echo "no peaks for ${tf}" >&2; exit 0
    fi
    MM="${UNIFIED}/${tf}/mm"
    if [[ ! -f "${MM}/matrix.mtx.gz" ]]; then
      echo "no raw mm for ${tf}" >&2; exit 0
    fi
    echo "[$(date)] export peak mm for Cicero  peaks=${PEAKS}"
    conda activate "${PY_ENV}"
    python "${COB}/scripts/export_tf_cicero_mm.py" \
      --tf "${TF}" --peaks "${PEAKS}" --mm-dir "${MM}" \
      --out-mm "${CWD}/mm" --max-peaks "${MAX_PEAKS}"

    echo "[$(date)] Cicero CDS + run_cicero window=${MAX_DIST}"
    conda activate "${CICERO_ENV}"
    module unload r 2>/dev/null || true
    cd "${CICERO_ROOT}"
    Rscript scripts/check_env.R
    Rscript scripts/02_build_cds.R --config "${CFG}" --work-dir "${CWD}"
    Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}"
    cd "${XULAB}"
  fi
  echo "[$(date)] Cicero conns -> cobinding edges (dedup unordered pairs)"
  conda activate "${PY_ENV}"
  python "${COB}/scripts/cicero_conns_to_edges.py" \
    --tf "${TF}" --conns "${CONNS}" --out-dir "${WORK}" \
    --chromhmm "${CHROMHMM}" --max-dist "${MAX_DIST}" \
    --coaccess-min "${COACCESS_MIN}"
  PDC="${WORK}/${TF}.generated.pdc"
  # pairs already score-filtered; keep all in the graph
  CLUSTER_QVAL=1
  CLUSTER_COMIN=0
fi

echo "[$(date)] cluster_peaks TF=${TF}"
conda activate "${PY_ENV}"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${EDGES}" --out-dir "${RESULTS}/${TF}" \
  --qval "${CLUSTER_QVAL}" --coaccess-min "${CLUSTER_COMIN}" --min-support 1 \
  --min-clique "${MIN_CLIQUE}" --share "${SHARE}" --plot \
  --net-min-comp "${NET_MIN_COMP}" --net-big "${NET_BIG}" \
  --max-dist "${MAX_DIST}"

if [[ -n "${PDC:-}" && -s "${PDC}" && -s "${CHROMHMM}" ]]; then
  echo "[$(date)] distal_chromhmm TF=${TF}"
  python "${COB}/scripts/distal_chromhmm_composition.py" \
    --conns "${PDC}" --chromhmm "${CHROMHMM}" \
    --out-dir "${RESULTS}/${TF}" --tf "${TF}" --qval "${CLUSTER_QVAL}"
fi
echo "[$(date)] done -> ${RESULTS}/${TF}"
