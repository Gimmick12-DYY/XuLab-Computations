#!/bin/bash
#SBATCH --job-name=rbbp4_wangk10
#SBATCH --output=rbbp4_wang_k10-%j.out
#SBATCH --error=rbbp4_wang_k10-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# Sidecar: Cicero on the lab RBBP4 pmat with settings that match HIS conns
# (all cells, k=10, 1 Mb, seed 2020). Does NOT touch gold RBBP4 results/edges
# or anything under /vast/som/xujie_lab/.
#
#   sbatch cobinding/slurm/run_rbbp4_wang_k10.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
TAG="${TAG:-RBBP4_wang_k10}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"
CFG="${CFG:-${COB}/configs/cicero_renlab.yaml}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${GOLD_EDGES:-${COB}/work/RBBP4.peak_edges.tsv}"
GOLD_SUMMARY="${GOLD_SUMMARY:-${COB}/results/RBBP4/summary.txt}"
SEL="${SEL:-${DATA_DIR}/TF.RBBP4.fitConns.res.sel}"
LAB_PMAT_SRC="${LAB_PMAT_SRC:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/pmat/seurat/RBBP4.pmat.mtx.rds}"
LAB_CONNS_SRC="${LAB_CONNS_SRC:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds}"
PERM_PARA="${PERM_PARA:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt}"
PMAT="${PMAT:-${DATA_DIR}/RBBP4.pmat.mtx.rds}"
SRC_MM="${SRC_MM:-${COB}/work/RBBP4_labpmat/cicero/mm}"
DO_SHUFFLE="${DO_SHUFFLE:-1}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"

export TMPDIR="${WORK}/tmp"
mkdir -p "${WORK}/cicero" "${RESULTS}" "${TMPDIR}"
echo "[$(date)] sidecar ${TAG}  cfg=${CFG}  work=${WORK}"

if [[ ! -s "${PMAT}" ]]; then
  [[ -s "${LAB_PMAT_SRC}" ]] || { echo "lab RBBP4 pmat not found: ${LAB_PMAT_SRC}" >&2; exit 1; }
  echo "[$(date)] copy lab RBBP4.pmat -> ${PMAT} (lab share not modified)"
  cp -n "${LAB_PMAT_SRC}" "${PMAT}"
fi

CWD="${WORK}/cicero"
CONNS="${CWD}/cicero/coaccess.tsv.gz"

if [[ -d "${SRC_MM}" && -s "${SRC_MM}/matrix.mtx.gz" ]]; then
  echo "[$(date)] reuse mm from ${SRC_MM}"
  ln -sfn "${SRC_MM}" "${CWD}/mm"
else
  echo "[$(date)] export lab RBBP4 pmat -> Cicero mm (all cells, min-cells-per-peak=1)"
  conda activate "${CICERO_ENV}"
  Rscript "${COB}/scripts/export_pmat_tf_mm.R" \
    --pmat "${PMAT}" --out-mm "${CWD}/mm" \
    --min-cells-per-peak 1 --min-peaks-per-cell 0
fi

echo "[$(date)] rebuild CDS (seed 2020, all cells) + run_cicero k=10 window=1Mb"
conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"
Rscript scripts/check_env.R
Rscript scripts/02_build_cds.R --config "${CFG}" --work-dir "${CWD}"
Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}"
cd "${XULAB}"

echo "[$(date)] cicero_info:"
cat "${CWD}/cicero/cicero_info.tsv"

echo "[$(date)] compare raw conns to lab RBBP4.conns.rds (read-only)"
if [[ -s "${LAB_CONNS_SRC}" && -s "${CONNS}" ]]; then
  Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
    --lab-conns "${LAB_CONNS_SRC}" \
    --ours "${CONNS}" \
    --out "${RESULTS}/compare_to_lab_conns.txt"
fi

echo "[$(date)] score our conns with his perm Gaussian para (p<=0.05 like .sel)"
conda activate "${PY_ENV}"
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
  echo "[$(date)] genbsp null (rebuild LSI+UMAP on bootstrap matrix)"
  conda activate "${CICERO_ENV}"
  cd "${CICERO_ROOT}"
  Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${CWD}" --shuffle genbsp
  cd "${XULAB}"
  OUR_PARA="${CWD}/cicero_shuf/shuffle.para.txt"
  echo "[$(date)] our genbsp para:"
  cat "${OUR_PARA}"
  echo "lab perm para:"
  cat "${PERM_PARA}"
  TAG_SHUF="${TAG}_genbsp"
  WORK_SHUF="${COB}/work/${TAG_SHUF}"
  RESULTS_SHUF="${COB}/results/${TAG_SHUF}"
  mkdir -p "${WORK_SHUF}" "${RESULTS_SHUF}"
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
  echo "---- our genbsp para vs gold ----"
  cat "${RESULTS_SHUF}/compare_to_gold.txt"
fi

echo "[$(date)] done -> ${RESULTS}"
[[ -s "${RESULTS}/compare_to_lab_conns.txt" ]] && cat "${RESULTS}/compare_to_lab_conns.txt"
echo "---- lab perm para p-values ----"
cat "${RESULTS}/compare_to_gold.txt"
