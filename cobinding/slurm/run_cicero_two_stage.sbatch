#!/bin/bash
#SBATCH --job-name=cob_2stage
#SBATCH --output=cob_2stage-%A_%a.out
#SBATCH --error=cob_2stage-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL
#
# Two-stage Cicero reproducibility filter.
# Motivation: single-round Cicero coordinated-binding pairs barely overlap Hi-C
# loops -> many may be spurious. So: re-run Cicero RESTRICTED to the peaks that
# appear in round-1's called pairs, call pairs again, and keep only pairs
# significant in BOTH rounds (reproducible = more likely real).
#
# NON-DESTRUCTIVE: reuses round-1 outputs read-only; writes only to
#   work/<TF>/cicero_stage2/   and   results/<TF>/stage2/.
# Requires round 1 first (run_cicero_cobinding.sbatch -> work/<TF>.peak_edges.tsv).
#
# RBBP4: do NOT use TF1000cells.pmat (different peak IDs). Use the lab TF-specific
# matrix whose rownames are the MACS peaks in
#   /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks/TF.RBBP4_peaks.q005.rmdup.tssUpDn1k.bed
#   (pmat: .../TAGATG-293T/combined/Cell1000/pmat/seurat/RBBP4.pmat.mtx.rds).
#
#   TF=RBBP4 sbatch cobinding/slurm/run_cicero_two_stage.sbatch
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
WORK="${COB_WORK:-${COB}/work}"
RESULTS="${COB_RESULTS:-${COB}/results}"
UNIFIED="${UNIFIED:-${XULAB}/unified/work}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
CFG="${CFG:-${COB}/configs/cicero_renlab.yaml}"
# default is the pooled TF1000cells pmat; RBBP4 overrides to the lab TF pmat below
PMAT="${PMAT:-}"
LAB_RBBP4_PMAT_SRC="${LAB_RBBP4_PMAT_SRC:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/pmat/seurat/RBBP4.pmat.mtx.rds}"
LAB_RBBP4_PEAKS="${LAB_RBBP4_PEAKS:-/vast/som/xujie_lab/TAGATG-293T/bigwig/peaks/TF.RBBP4_peaks.q005.rmdup.tssUpDn1k.bed}"
SHUFFLE_MODE="${SHUFFLE_MODE:-genbsp}"
QVAL="${QVAL:-0.05}"
MAX_DIST="${MAX_DIST:-1000000}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"; SHARE="${SHARE:-2}"
NET_MIN_COMP="${NET_MIN_COMP:-3}"; NET_BIG="${NET_BIG:-10}"
REBUILD="${REBUILD:-0}"
# optional Hi-C loop-overlap check (round1 vs reproducible): set all three layers
LOOP_CORE="${LOOP_CORE:-}"; LOOP_MID="${LOOP_MID:-}"; LOOP_LOOSE="${LOOP_LOOSE:-}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${TF:-}" ]]; then TF_ARR=("${TF}"); IDX=0; else
  mapfile -t TF_ARR < <("${PY_ENV}/bin/python" "${COB}/scripts/list_cobinding_tfs.py")
  IDX="${SLURM_ARRAY_TASK_ID:-0}"; fi
(( IDX >= 0 && IDX < ${#TF_ARR[@]} )) || { echo "idx ${IDX} OOR" >&2; exit 0; }
TF="${TF_ARR[$IDX]}"; tf="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"

EDGES1="${WORK}/${TF}.peak_edges.tsv"
[[ -s "${EDGES1}" ]] || { echo "no round-1 edges ${EDGES1}; run round 1 first" >&2; exit 1; }

# Peak matrix: RBBP4 uses lab TF-specific pmat (exact MACS peak IDs). Other TFs
# still use pooled TF1000cells + barcode subset unless PMAT is overridden.
BARCODE_ARGS=()
if [[ "${TF}" == "RBBP4" ]]; then
  PMAT="${PMAT:-${DATA_DIR}/RBBP4.pmat.mtx.rds}"
  if [[ ! -s "${PMAT}" ]]; then
    [[ -s "${LAB_RBBP4_PMAT_SRC}" ]] || { echo "lab RBBP4 pmat missing: ${LAB_RBBP4_PMAT_SRC}" >&2; exit 1; }
    echo "[$(date)] copy lab RBBP4.pmat -> ${PMAT} (vast share not modified)"
    cp -n "${LAB_RBBP4_PMAT_SRC}" "${PMAT}"
  fi
  echo "[$(date)] RBBP4 peaks (read-only): ${LAB_RBBP4_PEAKS}"
else
  PMAT="${PMAT:-${DATA_DIR}/TF1000cells.pmat.mtx.rds}"
  BARCODES="${UNIFIED}/${tf}/mm/barcodes.tsv.gz"
  [[ -s "${BARCODES}" ]] || { echo "no barcodes ${BARCODES}" >&2; exit 1; }
  BARCODE_ARGS=(--barcodes "${BARCODES}")
fi
[[ -s "${PMAT}" ]] || { echo "no pmat ${PMAT}" >&2; exit 1; }

CWD2="${WORK}/${TF}/cicero_stage2"
RES2="${RESULTS}/${TF}/stage2"
CONNS2="${CWD2}/cicero/coaccess.tsv.gz"
EDGES2="${CWD2}/${TF}.peak_edges.tsv"
mkdir -p "${CWD2}" "${RES2}"
echo "[$(date)] TF=${TF} two-stage; round1=${EDGES1}; pmat=${PMAT}"

# ---- 1. peaks from round-1 called pairs (q<=QVAL) ---------------------------
conda activate "${PY_ENV}"
python "${COB}/scripts/extract_pairs_peaks.py" \
  --edges "${EDGES1}" --qval "${QVAL}" --out "${CWD2}/stage1_peaks.txt"

# ---- 2. restricted mm + Cicero round 2 (+ genbsp null) ----------------------
if [[ "${REBUILD}" == "1" || ! -s "${CONNS2}" ]]; then
  conda activate "${CICERO_ENV}"
  Rscript "${COB}/scripts/export_pmat_tf_mm.R" \
    --pmat "${PMAT}" "${BARCODE_ARGS[@]}" --regions "${CWD2}/stage1_peaks.txt" \
    --out-mm "${CWD2}/mm" --min-cells-per-peak 1
  module unload r 2>/dev/null || true
  cd "${CICERO_ROOT}"
  Rscript scripts/check_env.R
  Rscript scripts/02_build_cds.R  --config "${CFG}" --work-dir "${CWD2}"
  Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD2}"
  Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD2}" --shuffle "${SHUFFLE_MODE}"
  cd "${XULAB}"
fi

# ---- 3. call pairs round 2 (same shuffle-Gaussian significance) -------------
conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf "${TF}" --conns "${CONNS2}" --out-dir "${CWD2}" \
  --chromhmm "${CHROMHMM}" --max-dist "${MAX_DIST}" --coaccess-min -1 \
  --pval-method shuffle --shuffle-para "${CWD2}/cicero_shuf/shuffle.para.txt"

# ---- 4. reproducible pairs = round1 INTERSECT round2 ------------------------
REPRO="${RES2}/${TF}.reproducible_pairs.tsv"
python "${COB}/scripts/intersect_pair_edges.py" \
  --a "${EDGES1}" --b "${EDGES2}" --qval "${QVAL}" --out "${REPRO}" | tee "${RES2}/intersect.txt"

# ---- 5. cluster on the reproducible set -------------------------------------
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${REPRO}" --out-dir "${RES2}" \
  --qval 1 --coaccess-min 0 --min-support 1 \
  --min-clique "${MIN_CLIQUE}" --share "${SHARE}" --plot \
  --net-min-comp "${NET_MIN_COMP}" --net-big "${NET_BIG}" --max-dist "${MAX_DIST}"

# ---- 6. optional: Hi-C loop overlap, round1 vs reproducible -----------------
if [[ -s "${LOOP_CORE}" && -s "${LOOP_MID}" && -s "${LOOP_LOOSE}" ]]; then
  echo "[$(date)] loop overlap: round1"
  python "${XULAB}/hic/scripts/compare_loops_to_cicero_pairs.py" \
    --core "${LOOP_CORE}" --mid "${LOOP_MID}" --loose "${LOOP_LOOSE}" \
    --edges "${EDGES1}" --qval "${QVAL}" --out-dir "${RES2}/loops_round1" || true
  echo "[$(date)] loop overlap: reproducible"
  python "${XULAB}/hic/scripts/compare_loops_to_cicero_pairs.py" \
    --core "${LOOP_CORE}" --mid "${LOOP_MID}" --loose "${LOOP_LOOSE}" \
    --edges "${REPRO}" --qval 1 --out-dir "${RES2}/loops_repro" || true
fi
echo "[$(date)] done -> ${RES2}"
