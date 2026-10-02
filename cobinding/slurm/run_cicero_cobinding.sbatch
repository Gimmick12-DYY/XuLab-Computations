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
# Cobinding's own Cicero runner (CDS + run_cicero). The imputation tree is imputation_legacy/Cicero.
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
WORK="${COB_WORK:-${COB}/work}"
RESULTS="${COB_RESULTS:-${COB}/results}"
UNIFIED="${UNIFIED:-${XULAB}/unified/work}"
PEAKS_IMP="${PEAKS_IMP:-${XULAB}/tf_complex/work/peaks_imputed}"
PEAKS_RAW="${PEAKS_RAW:-${XULAB}/tf_complex/work/peaks}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
CFG="${CFG:-${COB}/configs/cicero_1mb.yaml}"
# Fragment-called peak matrix (peaks x cells, all TFs) from the colleague. Cicero
# runs on THIS (correct, fragment-resolution) universe, subset to the TF's cells --
# not the 1 kb-bin synthetic peaks that export_tf_cicero_mm.py used to build.
PMAT="${PMAT:-${DATA_DIR}/TF1000cells.pmat.mtx.rds}"
MIN_CELLS_PER_PEAK="${MIN_CELLS_PER_PEAK:-2}"
MIN_PEAKS_PER_CELL="${MIN_PEAKS_PER_CELL:-0}"
IGNORE_FITCONNS="${IGNORE_FITCONNS:-0}"   # 1 = run our pmat Cicero even if lab fitConns exist (validate vs gold)

QVAL="${QVAL:-0.05}"
COACCESS_MIN="${COACCESS_MIN:-0.05}"
MAX_DIST="${MAX_DIST:-1000000}"
MAX_PEAKS="${MAX_PEAKS:-20000}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"
SHARE="${SHARE:-2}"
NET_MIN_COMP="${NET_MIN_COMP:-3}"
NET_BIG="${NET_BIG:-10}"
REBUILD_CICERO="${REBUILD_CICERO:-0}"
# Significance for our own Cicero (no lab fitConns): shuffle = Ren-lab null
# (Li 2021/Zu 2023) -- permute peaks per cell, refit Cicero, fit Gaussian to the
# null coaccess, test real scores (BH FDR). ttest = coaccess correlation t-test.
SIG_METHOD="${SIG_METHOD:-shuffle}"
SHUFFLE_MODE="${SHUFFLE_MODE:-colshuf}"

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

if [[ "${HAVE_FIT}" == "1" && "${IGNORE_FITCONNS}" != "1" ]]; then
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
    [[ -s "${PMAT}" ]] || { echo "no pmat: ${PMAT}" >&2; exit 0; }
    BARCODES="${UNIFIED}/${tf}/mm/barcodes.tsv.gz"
    if [[ ! -s "${BARCODES}" ]]; then
      echo "no per-TF barcodes for ${tf} at ${BARCODES}" >&2; exit 0
    fi
    echo "[$(date)] subset fragment-called pmat to ${TF} cells -> Cicero mm"
    conda activate "${CICERO_ENV}"     # R/Matrix to read the pmat RDS + write mm
    Rscript "${COB}/scripts/export_pmat_tf_mm.R" \
      --pmat "${PMAT}" --barcodes "${BARCODES}" --out-mm "${CWD}/mm" \
      --min-cells-per-peak "${MIN_CELLS_PER_PEAK}" \
      --min-peaks-per-cell "${MIN_PEAKS_PER_CELL}"

    echo "[$(date)] Cicero CDS + run_cicero window=${MAX_DIST}"
    conda activate "${CICERO_ENV}"
    module unload r 2>/dev/null || true
    cd "${CICERO_ROOT}"
    Rscript scripts/check_env.R
    Rscript scripts/02_build_cds.R --config "${CFG}" --work-dir "${CWD}"
    Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}"
    if [[ "${SIG_METHOD}" == "shuffle" ]]; then
      echo "[$(date)] Cicero shuffle null (${SHUFFLE_MODE}) -> cicero_shuf/shuffle.para.txt"
      Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}" --shuffle "${SHUFFLE_MODE}"
    fi
    cd "${XULAB}"
  fi
  echo "[$(date)] Cicero conns -> cobinding edges (sig=${SIG_METHOD})"
  conda activate "${PY_ENV}"
  SIG_ARGS=(); COMIN="${COACCESS_MIN}"
  if [[ "${SIG_METHOD}" == "shuffle" ]]; then
    SHUF_PARA="${CWD}/cicero_shuf/shuffle.para.txt"
    [[ -s "${SHUF_PARA}" ]] || { echo "missing shuffle para ${SHUF_PARA}; rerun with REBUILD_CICERO=1" >&2; exit 1; }
    SIG_ARGS=(--pval-method shuffle --shuffle-para "${SHUF_PARA}")
    COMIN=-1; CLUSTER_QVAL="${QVAL}"     # BH over all pairs vs null; filter on real FDR
  else
    CLUSTER_QVAL=1                        # ttest path: coaccess-filtered, keep all in graph
  fi
  python "${COB}/scripts/cicero_conns_to_edges.py" \
    --tf "${TF}" --conns "${CONNS}" --out-dir "${WORK}" \
    --chromhmm "${CHROMHMM}" --max-dist "${MAX_DIST}" \
    --coaccess-min "${COMIN}" "${SIG_ARGS[@]}"
  PDC="${WORK}/${TF}.generated.pdc"
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
