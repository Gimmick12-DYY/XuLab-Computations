#!/bin/bash
#SBATCH --job-name=cobinding
#SBATCH --output=cobinding-%A_%a.out
#SBATCH --error=cobinding-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Higher-order co-binding clusters from Cicero-style cis pairs (≤ 1 Mb).
# Same plots as RBBP4 for every TF:
#   clique_sizes / module_sizes / tf_peak_network{,_clusters,_cliques}
#   clique_chromosomes / distal_chromhmm_composition
#
#   # all TFs with imputed (or raw) peaks
#   N=$(python cobinding/scripts/list_cobinding_tfs.py | wc -l)
#   sbatch --array=0-$((N-1))%12 cobinding/slurm/run_cobinding.sbatch
#
#   TF=RBBP4 sbatch cobinding/slurm/run_cobinding.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
WORK="${COB_WORK:-${COB}/work}"
RESULTS="${COB_RESULTS:-${COB}/results}"
UNIFIED="${UNIFIED:-${XULAB}/unified/work}"
PEAKS_IMP="${PEAKS_IMP:-${XULAB}/tf_complex/work/peaks_imputed}"
PEAKS_RAW="${PEAKS_RAW:-${XULAB}/tf_complex/work/peaks}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"

if [[ -z "${GLOB:-}" ]]; then
  GLOB='TF.{tf}.fitConns.res.*'
fi
REBUILD="${REBUILD:-1}"
Q_KEEP="${Q_KEEP:-1.0}"
QVAL="${QVAL:-0.05}"
COACCESS_MIN="${COACCESS_MIN:-0.0}"
MIN_SUPPORT="${MIN_SUPPORT:-1}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"
SHARE="${SHARE:-2}"
NET_MIN_COMP="${NET_MIN_COMP:-3}"
NET_BIG="${NET_BIG:-10}"
MAX_DIST="${MAX_DIST:-1000000}"
MAX_PEAKS="${MAX_PEAKS:-20000}"

CONDA_ENV="${CONDA_ENV:-cobinding}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# TF list: explicit TF=, else array over every TF with a peak bed + matrix
if [[ -n "${TF:-}" ]]; then
  TF_ARR=("${TF}")
  IDX=0
else
  mapfile -t TF_ARR < <(python "${COB}/scripts/list_cobinding_tfs.py")
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
fi
if (( IDX < 0 || IDX >= ${#TF_ARR[@]} )); then
  echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2
  exit 0
fi
TF="${TF_ARR[$IDX]}"
tf="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
echo "[$(date)] TF=${TF} idx=${IDX} max_dist=${MAX_DIST}"

mkdir -p "${WORK}" "${RESULTS}/${TF}"
EDGES="${WORK}/${TF}.peak_edges.tsv"
PDC=""
for cand in \
  "${DATA_DIR}/TF.${TF}.fitConns.res.pdc" \
  "${DATA_DIR}/TF.${tf}.fitConns.res.pdc" \
  "${WORK}/${TF}.generated.pdc"
do
  [[ -s "${cand}" ]] && PDC="${cand}" && break
done

HAVE_FIT=0
for cand in "${DATA_DIR}/TF.${TF}.fitConns.res.sel" "${DATA_DIR}/TF.${tf}.fitConns.res.sel" \
            "${DATA_DIR}/TF.${TF}.fitConns.res.pdc" "${DATA_DIR}/TF.${tf}.fitConns.res.pdc"; do
  [[ -s "${cand}" ]] && HAVE_FIT=1 && break
done

if [[ "${REBUILD}" == "1" || ! -s "${EDGES}" ]]; then
  if [[ "${HAVE_FIT}" == "1" ]]; then
    echo "[$(date)] build_peak_graph from fitConns (Cicero) TF=${TF}"
    python "${COB}/scripts/build_peak_graph.py" \
      --data-dir "${DATA_DIR}" --tf "${TF}" --glob "${GLOB}" \
      --out-dir "${WORK}" --q-keep "${Q_KEEP}" --max-dist "${MAX_DIST}"
  else
    PEAKS=""
    if [[ -s "${PEAKS_IMP}/${tf}.bed" ]]; then
      PEAKS="${PEAKS_IMP}/${tf}.bed"
    elif [[ -s "${PEAKS_RAW}/${tf}.bed" ]]; then
      PEAKS="${PEAKS_RAW}/${tf}.bed"
    else
      echo "no peaks for ${tf}" >&2
      exit 0
    fi
    MAT=""
    if [[ -f "${UNIFIED}/${tf}/impute/matrix_csr.npz" ]]; then
      MAT="${UNIFIED}/${tf}/impute"
    elif [[ -f "${UNIFIED}/${tf}/mm/matrix.mtx.gz" ]]; then
      MAT="${UNIFIED}/${tf}/mm"
    else
      echo "no matrix for ${tf}" >&2
      exit 0
    fi
    echo "[$(date)] build_pairs_from_matrix TF=${TF} peaks=${PEAKS} mat=${MAT}"
    python "${COB}/scripts/build_pairs_from_matrix.py" \
      --tf "${TF}" --peaks "${PEAKS}" --mat-dir "${MAT}" \
      --out-dir "${WORK}" --chromhmm "${CHROMHMM}" \
      --max-dist "${MAX_DIST}" --max-peaks "${MAX_PEAKS}" --qval "${QVAL}"
    PDC="${WORK}/${TF}.generated.pdc"
  fi
else
  echo "[$(date)] reuse ${EDGES}"
fi

echo "[$(date)] cluster_peaks TF=${TF} (q<=${QVAL} max_dist=${MAX_DIST})"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${EDGES}" --out-dir "${RESULTS}/${TF}" \
  --qval "${QVAL}" --coaccess-min "${COACCESS_MIN}" --min-support "${MIN_SUPPORT}" \
  --min-clique "${MIN_CLIQUE}" --share "${SHARE}" --plot \
  --net-min-comp "${NET_MIN_COMP}" --net-big "${NET_BIG}" \
  --max-dist "${MAX_DIST}"

if [[ -n "${PDC}" && -s "${PDC}" && -s "${CHROMHMM}" ]]; then
  echo "[$(date)] distal_chromhmm TF=${TF}"
  python "${COB}/scripts/distal_chromhmm_composition.py" \
    --conns "${PDC}" --chromhmm "${CHROMHMM}" \
    --out-dir "${RESULTS}/${TF}" --tf "${TF}" --qval "${QVAL}"
else
  echo "[$(date)] skip distal_chromhmm (pdc=${PDC:-none})"
fi
echo "[$(date)] done -> ${RESULTS}/${TF}"
