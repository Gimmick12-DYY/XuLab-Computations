#!/bin/bash
#SBATCH --job-name=rbbp4_peak2fit
#SBATCH --output=rbbp4_peak2fit-%j.out
#SBATCH --error=rbbp4_peak2fit-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#
# End-to-end RBBP4 reconstruction:
#   peak BED + selected-cell fragment dataset
#     -> peak matrix -> LSI/UMAP -> Cicero -> genBsp null
#     -> fitConns p/FDR -> selected pairs -> cliques.
#
# The /vast source tree is read-only. All outputs are sidecars.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
SRC="${SRC:-/vast/som/xujie_lab/TAGATG-293T}"
TAG="${TAG:-RBBP4_from_peaks}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"

PEAKS="${PEAKS:-${SRC}/bigwig/peaks/TF.RBBP4_peaks.q005.rmdup.narrowPeak}"
DATASET="${DATASET:-${SRC}/combined/Cell1000/RBBP4/_dataset.h5ads}"
ANNDATA_DIR="${ANNDATA_DIR:-${SRC}/combined/Cell1000/RBBP4/anndatas}"
REFERENCE_PMAT="${REFERENCE_PMAT:-${SRC}/combined/Cell1000/pmat/RBBP4.pmat.h5ad}"
LAB_CONNS="${LAB_CONNS:-${SRC}/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds}"
GOLD_EDGES="${GOLD_EDGES:-${COB}/work/RBBP4.peak_edges.tsv}"
GOLD_SUMMARY="${GOLD_SUMMARY:-${COB}/results/RBBP4/summary.txt}"
SEL="${SEL:-${XULAB}/data/TF.RBBP4.fitConns.res.sel}"
CHROMHMM="${CHROMHMM:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"

SNAP_ENV="${SNAP_ENV:-/work/users/d/y/dyy12/conda/envs/snapatac2}"
PY_ENV="${PY_ENV:-/work/users/d/y/dyy12/conda/envs/cobinding}"
CICERO_ENV="${CICERO_ENV:-cicero}"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
REAL_CFG="${REAL_CFG:-${COB}/configs/cicero_renlab.yaml}"
NULL_CFG="${NULL_CFG:-${COB}/configs/cicero_wang_null.yaml}"

REBUILD_PMAT="${REBUILD_PMAT:-1}"
REBUILD_CICERO="${REBUILD_CICERO:-1}"

for p in "${PEAKS}" "${DATASET}" "${REFERENCE_PMAT}" "${LAB_CONNS}"; do
  [[ -s "${p}" ]] || { echo "missing ${p}" >&2; exit 1; }
done
[[ -d "${ANNDATA_DIR}" ]] || { echo "missing ${ANNDATA_DIR}" >&2; exit 1; }
[[ -x "${SNAP_ENV}/bin/python" ]] || {
  echo "missing SnapATAC2 env ${SNAP_ENV}; run 00_setup_snapatac2.sbatch" >&2
  exit 1
}

mkdir -p "${WORK}/mm" "${WORK}/tmp" "${RESULTS}"
export TMPDIR="${WORK}/tmp"
source "$(conda info --base)/etc/profile.d/conda.sh"

PMAT="${WORK}/RBBP4.pmat.from_peaks.h5ad"
DATASET_LOCAL="${WORK}/RBBP4.dataset.local.h5ads"
CONNS="${WORK}/cicero/coaccess.tsv.gz"
NULL_PARA="${WORK}/cicero_shuf/shuffle.para.txt"
FITCONNS="${WORK}/RBBP4.perm.fitConns.res.txt.gz"

if [[ "${REBUILD_PMAT}" == "1" || ! -s "${PMAT}" ]]; then
  echo "[$(date)] stage 1: peak BED + selected fragments -> peak matrix"
  # read_dataset(..., adata_files_update=) rewrites stale component links.
  # Work on a local metadata copy; never modify the lab's /vast source.
  cp -f "${DATASET}" "${DATASET_LOCAL}"
  "${SNAP_ENV}/bin/python" "${COB}/scripts/rebuild_rbbp4_peak_matrix.py" \
    --dataset "${DATASET_LOCAL}" \
    --anndata-dir "${ANNDATA_DIR}" \
    --peaks "${PEAKS}" \
    --out-h5ad "${PMAT}" \
    --out-mm-dir "${WORK}/mm" \
    --reference-h5ad "${REFERENCE_PMAT}"
fi

if [[ "${REBUILD_CICERO}" == "1" || ! -s "${CONNS}" || ! -s "${NULL_PARA}" ]]; then
  echo "[$(date)] stage 2: exact peak matrix -> CDS/LSI/UMAP"
  conda activate "${CICERO_ENV}"
  module unload r 2>/dev/null || true
  cd "${CICERO_ROOT}"
  Rscript scripts/check_env.R
  Rscript scripts/02_build_cds.R --config "${REAL_CFG}" --work-dir "${WORK}"

  echo "[$(date)] stage 3: real Cicero (k=10, 1 Mb, seed=2020)"
  Rscript scripts/03_run_cicero.R --config "${REAL_CFG}" --work-dir "${WORK}"

  echo "[$(date)] stage 4: Wang/Ren genBsp null (default k=50)"
  Rscript scripts/03_run_cicero.R \
    --config "${NULL_CFG}" --work-dir "${WORK}" --shuffle genbsp
  cd "${XULAB}"
fi

echo "[$(date)] compare raw re-run Cicero to Wang raw conns"
conda activate "${CICERO_ENV}"
Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
  --lab-conns "${LAB_CONNS}" \
  --ours "${CONNS}" \
  --out "${RESULTS}/compare_raw_conns_to_wang.txt"

echo "[$(date)] stage 5: raw conns + reconstructed null -> fitConns pairs"
conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method shuffle --shuffle-para "${NULL_PARA}" \
  --scored-all-out "${FITCONNS}" \
  --sel "${SEL}"

echo "[$(date)] stage 6: cluster reconstructed pairs"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" \
  --annot "${WORK}/RBBP4.peak_annot.tsv" \
  --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --plot \
  --net-min-comp 3 --net-big 10 --max-dist 1000000

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" \
  --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
  --gold-summary "${GOLD_SUMMARY}" \
  --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" \
  --qval 0.05 --coaccess-min 0

echo "==== peak matrix ===="
cat "${PMAT%.h5ad}.compare_to_reference.txt"
echo "==== raw Cicero conns ===="
cat "${RESULTS}/compare_raw_conns_to_wang.txt"
echo "==== fitConns pairs ===="
cat "${RESULTS}/compare_to_gold.txt"
echo "[$(date)] complete -> ${RESULTS}"
