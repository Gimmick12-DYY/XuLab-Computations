#!/bin/bash
#SBATCH --job-name=rbbp4_k50cap
#SBATCH --output=rbbp4_labpmat_k50cap-%j.out
#SBATCH --error=rbbp4_labpmat_k50cap-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# Reuse the RBBP4_labpmat CDS (same peaks/cells/UMAP) and re-run
# make_cicero_cds with the 5000-iteration cap lifted (20000), then
# run_cicero, then his shuffle Gaussian p-values.
# Optionally a UMAP-permuted shuffle Cicero for our own mu/sd.
# Does not touch gold RBBP4 or /vast/som/xujie_lab/.
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
SRC="${COB}/work/RBBP4_labpmat/cicero"
TAG="${TAG:-RBBP4_labpmat_k50cap}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"
CFG="${COB}/configs/cicero_1mb_labpmat_k50cap.yaml"
CFG_SHUF="${COB}/configs/cicero_1mb_labpmat_k50cap_shuf.yaml"
CHROMHMM="${CHROMHMM:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${COB}/work/RBBP4.peak_edges.tsv"
GOLD_SUMMARY="${COB}/results/RBBP4/summary.txt"
SEL="${XULAB}/data/TF.RBBP4.fitConns.res.sel"
LAB_CONNS_SRC="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds"
PERM_PARA="${PERM_PARA:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt}"
DO_SHUFFLE="${DO_SHUFFLE:-1}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"

[[ -s "${SRC}/cds/cds.rds" ]] || { echo "missing CDS ${SRC}/cds/cds.rds" >&2; exit 1; }

export TMPDIR="${WORK}/tmp"
mkdir -p "${WORK}/cicero" "${RESULTS}" "${TMPDIR}"
ln -sfn "${SRC}/mm" "${WORK}/cicero/mm"
ln -sfn "${SRC}/cds" "${WORK}/cicero/cds"

echo "[$(date)] k=50 max_iter=20000 Cicero on existing CDS  work=${WORK}"
conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"
Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${WORK}/cicero"
cd "${XULAB}"

CONNS="${WORK}/cicero/cicero/coaccess.tsv.gz"
echo "[$(date)] n_metacell:"
cat "${WORK}/cicero/cicero/cicero_info.tsv"

if [[ -s "${LAB_CONNS_SRC}" ]]; then
  Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
    --lab-conns "${LAB_CONNS_SRC}" --ours "${CONNS}" \
    --out "${RESULTS}/compare_to_lab_conns.txt"
fi

conda activate "${PY_ENV}"
# His p formula on our new scores; keep p<=0.05 like .sel; cluster q<=0.05.
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method shuffle --shuffle-para "${PERM_PARA}" \
  --sel "${SEL}"

python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --plot --net-min-comp 3 --net-big 10 --max-dist 1000000

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
  --gold-summary "${GOLD_SUMMARY}" --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0

if [[ "${DO_SHUFFLE}" == "1" ]]; then
  TAG_SHUF="${TAG}_shuf"
  WORK_SHUF="${COB}/work/${TAG_SHUF}"
  RESULTS_SHUF="${COB}/results/${TAG_SHUF}"
  export TMPDIR="${WORK_SHUF}/tmp"
  mkdir -p "${WORK_SHUF}/cicero" "${RESULTS_SHUF}" "${TMPDIR}"
  ln -sfn "${SRC}/mm" "${WORK_SHUF}/cicero/mm"
  ln -sfn "${SRC}/cds" "${WORK_SHUF}/cicero/cds"
  echo "[$(date)] shuffle UMAP Cicero (own mu/sd)  work=${WORK_SHUF}"
  conda activate "${CICERO_ENV}"
  cd "${CICERO_ROOT}"
  Rscript scripts/03_run_cicero.R --config "${CFG_SHUF}" --work-dir "${WORK_SHUF}/cicero"
  cd "${XULAB}"
  OUR_PARA="${WORK_SHUF}/cicero/cicero/shuffle.para.txt"
  echo "[$(date)] our shuffle para:"
  cat "${OUR_PARA}"
  conda activate "${PY_ENV}"
  python "${COB}/scripts/cicero_conns_to_edges.py" \
    --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK_SHUF}" \
    --chromhmm "${CHROMHMM}" --max-dist 1000000 \
    --coaccess-min -1 --max-p 0.05 \
    --pval-method shuffle --shuffle-para "${OUR_PARA}" \
    --sel "${SEL}"
  python "${COB}/scripts/cluster_peaks.py" \
    --edges "${WORK_SHUF}/RBBP4.peak_edges.tsv" --out-dir "${RESULTS_SHUF}" \
    --qval 0.05 --coaccess-min 0 --min-support 1 \
    --min-clique 3 --share 2 --plot --net-min-comp 3 --net-big 10 --max-dist 1000000
  python "${COB}/scripts/compare_rbbp4_repro.py" \
    --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK_SHUF}/RBBP4.peak_edges.tsv" \
    --gold-summary "${GOLD_SUMMARY}" --repro-summary "${RESULTS_SHUF}/summary.txt" \
    --out "${RESULTS_SHUF}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0
  echo "---- our-para shuffle null ----"
  cat "${RESULTS_SHUF}/compare_to_gold.txt"
fi

echo "[$(date)] done -> ${RESULTS}"
[[ -s "${RESULTS}/compare_to_lab_conns.txt" ]] && cat "${RESULTS}/compare_to_lab_conns.txt"
echo "---- lab perm para p-values ----"
cat "${RESULTS}/compare_to_gold.txt"
