#!/bin/bash
#SBATCH --job-name=ourrun_tf
#SBATCH --output=/work/users/d/y/dyy12/XuLab/cobinding/logs/ourrun-%A_%a.out
#SBATCH --error=/work/users/d/y/dyy12/XuLab/cobinding/logs/ourrun-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#
# Our Run, one TF per task. Same Cicero settings that produced the RBBP4
# "Our Run" column (20,613 pairs):
#   k=50, metacell iteration cap 10,000, seed 555, sample_num 100,
#   1 Mb window, then p<=0.05 against the same Gaussian null
#   (RBBP4.perm.fitConns.para.txt), then FDR<=0.05 cliques.
#
# Peaks are the MACS2 narrowPeak in bigwig/macs2/TF.<TF>/, kept at q <= 0.05
# using the q column in that file. Fragments are the treatment BED named in
# the MACS2 xls (bigwig/fragment/TF.<TF>.bed.gz); the cell barcode is column 4.
# RBBP4 is not in this array: its 1 Mb Our Run already exists.
#
# This job is the first Cicero run only (1 Mb). The 2 Mb search is a later
# job and is not started here. Loop metrics and the permutation run on the
# compute node after the 1 Mb pairs and cliques exist.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
SRC="/vast/som/xujie_lab/TAGATG-293T"
CICERO_ROOT="${COB}/cicero"
PERM_PARA="${SRC}/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt"
CHROMHMM="${XULAB}/data/HEK293T_chromHMM18.bed.gz"
TF_LIST="${COB}/configs/ourrun_tfs.txt"
PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

mapfile -t TF_ARR < "${TF_LIST}"
IDX="${SLURM_ARRAY_TASK_ID:?}"
TF="${TF_ARR[$IDX]}"
if [[ -z "${TF}" || "${TF}" == "RBBP4" ]]; then
  echo "refusing index ${IDX} TF='${TF}'" >&2
  exit 1
fi

PEAKS="${SRC}/bigwig/macs2/TF.${TF}/TF.${TF}_peaks.narrowPeak"
FRAGMENTS="${SRC}/bigwig/fragment/TF.${TF}.bed.gz"
WORK="${COB}/work/ourrun_k50/${TF}"
RES="${COB}/results/ourrun_k50/${TF}"
CWD="${WORK}/cicero"
mkdir -p "${WORK}/tmp" "${RES}" "${COB}/logs"
export TMPDIR="${WORK}/tmp"
echo "[$(date)] TF=${TF} idx=${IDX} peaks=${PEAKS}"
[[ -s "${PEAKS}" && -s "${FRAGMENTS}" ]] || { echo "missing peaks or fragments for ${TF}" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -s "${CWD}/mm/matrix.mtx.gz" ]]; then
  echo "[$(date)] peak matrix from MACS2 peaks + fragment BED"
  conda activate "${PY_ENV}"
  set +e
  python "${COB}/scripts/build_tf_peak_mm_from_macs2.py" \
    --tf "${TF}" \
    --peaks "${PEAKS}" \
    --fragments "${FRAGMENTS}" \
    --out-mm-dir "${CWD}/mm" \
    --max-q 0.05
  rc=$?
  set -e
  if [[ "${rc}" -eq 2 ]]; then
    echo "[$(date)] ${TF} has no q<=0.05 peaks; first Cicero run skipped"
    exit 0
  fi
  [[ "${rc}" -eq 0 ]] || exit "${rc}"
fi

write_cfg() {
  local dest="$1" window="$2"
  cat > "${dest}" <<EOF
paths:
  work_dir: ${CWD}
cicero:
  min_cells_per_peak: 1
  num_dim: 50
  k_metacell: 50
  metacell_max_iter: 10000
  window_bp: ${window}
  sample_num: 100
  max_cells: 0
  random_seed: 555
  shuffle: none
EOF
}

conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"

if [[ ! -s "${CWD}/cds/cds.rds" ]]; then
  echo "[$(date)] CDS / LSI / UMAP"
  write_cfg "${CWD}/config.yaml" 1000000
  Rscript scripts/02_build_cds.R --config "${CWD}/config.yaml" --work-dir "${CWD}"
fi

if [[ ! -s "${CWD}/cicero/coaccess.tsv.gz" ]]; then
  echo "[$(date)] Cicero 1 Mb"
  write_cfg "${CWD}/config.yaml" 1000000
  Rscript scripts/03_run_cicero.R --config "${CWD}/config.yaml" --work-dir "${CWD}"
fi

conda activate "${PY_ENV}"
cd "${XULAB}"
EDGES="${WORK}/${TF}.peak_edges.tsv"
if [[ ! -s "${EDGES}" ]]; then
  echo "[$(date)] score p<=0.05"
  python "${COB}/scripts/cicero_conns_to_edges.py" \
    --tf "${TF}" --conns "${CWD}/cicero/coaccess.tsv.gz" --out-dir "${WORK}" \
    --chromhmm "${CHROMHMM}" --max-dist 1000000 \
    --coaccess-min -1 --max-p 0.05 \
    --pval-method shuffle --shuffle-para "${PERM_PARA}"
fi

NEDGE=$(($(wc -l < "${EDGES}") - 1))
echo "[$(date)] 1 Mb selected pairs ${NEDGE}"
if (( NEDGE < 1 )); then
  echo "[$(date)] no selected pairs; cliques not built"
  exit 0
fi

if [[ ! -s "${RES}/cliques.tsv" ]]; then
  echo "[$(date)] cliques"
  python "${COB}/scripts/cluster_peaks.py" \
    --edges "${EDGES}" --out-dir "${RES}" \
    --qval 0.05 --coaccess-min 0 --min-support 1 \
    --min-clique 3 --share 2 --net-min-comp 3 --net-big 10 \
    --max-dist 1000000
fi

if [[ ! -s "${RES}/slide/permutation.tsv" ]]; then
  echo "[$(date)] loop metrics and permutation"
  python "${XULAB}/hic/scripts/fill_slide_loop_table.py" \
    --n-perm 500 --seed 1 \
    --label ourrun --edges "${EDGES}" --cliques "${RES}/cliques.tsv" \
    --out-dir "${RES}/slide"
fi

echo "[$(date)] done 1 Mb ${TF}"
