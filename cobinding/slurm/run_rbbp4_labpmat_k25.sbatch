#!/bin/bash
#SBATCH --job-name=rbbp4_k25
#SBATCH --output=rbbp4_labpmat_k25-%j.out
#SBATCH --error=rbbp4_labpmat_k25-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL
#
# Reuse the RBBP4_labpmat CDS (same peaks/cells/UMAP) and only re-run
# make_cicero_cds + run_cicero with k=25. Does not touch gold RBBP4 or
# /vast/som/xujie_lab/.
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
SRC="${COB}/work/RBBP4_labpmat/cicero"
TAG="${TAG:-RBBP4_labpmat_k25}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"
CFG="${COB}/configs/cicero_1mb_labpmat_k25.yaml"
CHROMHMM="${CHROMHMM:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${COB}/work/RBBP4.peak_edges.tsv"
GOLD_SUMMARY="${COB}/results/RBBP4/summary.txt"
LAB_CONNS_SRC="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"

[[ -s "${SRC}/cds/cds.rds" ]] || { echo "missing CDS ${SRC}/cds/cds.rds" >&2; exit 1; }

mkdir -p "${WORK}/cicero" "${RESULTS}"
# reuse mm + CDS (UMAP unchanged); only Cicero aggregation/k changes
ln -sfn "${SRC}/mm" "${WORK}/cicero/mm"
ln -sfn "${SRC}/cds" "${WORK}/cicero/cds"

echo "[$(date)] k=25 Cicero on existing CDS  work=${WORK}"
conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"
Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${WORK}/cicero"
cd "${XULAB}"

CONNS="${WORK}/cicero/cicero/coaccess.tsv.gz"
if [[ -s "${LAB_CONNS_SRC}" ]]; then
  Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
    --lab-conns "${LAB_CONNS_SRC}" --ours "${CONNS}" \
    --out "${RESULTS}/compare_to_lab_conns.txt"
fi

conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 --coaccess-min 0.05

python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" --out-dir "${RESULTS}" \
  --qval 1 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --plot --net-min-comp 3 --net-big 10 --max-dist 1000000

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
  --gold-summary "${GOLD_SUMMARY}" --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0.05

echo "[$(date)] done -> ${RESULTS}"
cat "${RESULTS}/compare_to_lab_conns.txt"
echo "----"
cat "${RESULTS}/compare_to_gold.txt"
