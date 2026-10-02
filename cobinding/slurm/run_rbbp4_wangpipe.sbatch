#!/bin/bash
#SBATCH --job-name=rbbp4_wangpipe
#SBATCH --output=rbbp4_wangpipe-%j.out
#SBATCH --error=rbbp4_wangpipe-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# Mimic Ren Lab / Dr. Wang Cicero p-values (Li 2021, Zu 2023):
#   real Cicero scores (reuse k50cap) + shuffled peak-by-cell Cicero
#   -> fitdistrplus Gaussian mu/sd -> BH FDR -> cluster q<=0.05.
# Does not touch gold RBBP4 or /vast/som/xujie_lab/.
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${CICERO_ROOT:-${COB}/cicero}"
SRC="${COB}/work/RBBP4_labpmat/cicero"
REAL_CONNS="${COB}/work/RBBP4_labpmat_k50cap/cicero/cicero/coaccess.tsv.gz"
CHROMHMM="${CHROMHMM:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${COB}/work/RBBP4.peak_edges.tsv"
GOLD_SUMMARY="${COB}/results/RBBP4/summary.txt"
SEL="${XULAB}/data/TF.RBBP4.fitConns.res.sel"
LAB_PERM_PARA="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt"
LAB_COL_PARA="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.colShuf.fitConns.para.txt"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

source "$(conda info --base)/etc/profile.d/conda.sh"
[[ -s "${SRC}/cds/cds.rds" ]] || { echo "missing CDS ${SRC}/cds/cds.rds" >&2; exit 1; }
[[ -s "${REAL_CONNS}" ]] || { echo "missing real Cicero table ${REAL_CONNS}" >&2; exit 1; }

score_and_cluster () {
  local para="$1" work="$2" results="$3" label="$4"
  mkdir -p "${work}" "${results}"
  echo "[$(date)] score real conns with ${label} para=${para}"
  conda activate "${PY_ENV}"
  python "${COB}/scripts/cicero_conns_to_edges.py" \
    --tf RBBP4 --conns "${REAL_CONNS}" --out-dir "${work}" \
    --chromhmm "${CHROMHMM}" --max-dist 1000000 \
    --coaccess-min -1 --max-p 0.05 \
    --pval-method shuffle --shuffle-para "${para}" \
    --sel "${SEL}"
  python "${COB}/scripts/cluster_peaks.py" \
    --edges "${work}/RBBP4.peak_edges.tsv" --out-dir "${results}" \
    --qval 0.05 --coaccess-min 0 --min-support 1 \
    --min-clique 3 --share 2 --plot --net-min-comp 3 --net-big 10 --max-dist 1000000
  python "${COB}/scripts/compare_rbbp4_repro.py" \
    --gold-edges "${GOLD_EDGES}" --repro-edges "${work}/RBBP4.peak_edges.tsv" \
    --gold-summary "${GOLD_SUMMARY}" --repro-summary "${results}/summary.txt" \
    --out "${results}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0
  echo "==== ${label} vs gold ===="
  cat "${results}/compare_to_gold.txt"
  echo "para:"
  cat "${para}"
}

run_shuffled_cicero () {
  local tag="$1" cfg="$2"
  local work="${COB}/work/${tag}"
  export TMPDIR="${work}/tmp"
  mkdir -p "${work}/cicero" "${TMPDIR}"
  ln -sfn "${SRC}/mm" "${work}/cicero/mm"
  ln -sfn "${SRC}/cds" "${work}/cicero/cds"
  echo "[$(date)] shuffled Cicero ${tag}  cfg=${cfg}"
  conda activate "${CICERO_ENV}"
  module unload r 2>/dev/null || true
  cd "${CICERO_ROOT}"
  Rscript scripts/03_run_cicero.R --config "${cfg}" --work-dir "${work}/cicero"
  cd "${XULAB}"
  cat "${work}/cicero/cicero/shuffle.para.txt"
}

echo "[$(date)] Wang/Ren pipeline on existing k50cap Cicero scores"

# 1) Dr. Wang's own perm para (already known to rebuild gold from HIS conns)
score_and_cluster \
  "${LAB_PERM_PARA}" \
  "${COB}/work/RBBP4_wangpipe_labperm" \
  "${COB}/results/RBBP4_wangpipe_labperm" \
  "lab-perm-para"

# 2) Our perm null (peak profiles shuffled across loci) + FDR on OUR scores
run_shuffled_cicero RBBP4_wangpipe_perm "${COB}/configs/cicero_1mb_wang_perm.yaml"
score_and_cluster \
  "${COB}/work/RBBP4_wangpipe_perm/cicero/cicero/shuffle.para.txt" \
  "${COB}/work/RBBP4_wangpipe_perm" \
  "${COB}/results/RBBP4_wangpipe_perm" \
  "our-perm"

# 3) Zu 2023 colShuf (cCREs shuffled within each cell)
run_shuffled_cicero RBBP4_wangpipe_colshuf "${COB}/configs/cicero_1mb_wang_colshuf.yaml"
score_and_cluster \
  "${COB}/work/RBBP4_wangpipe_colshuf/cicero/cicero/shuffle.para.txt" \
  "${COB}/work/RBBP4_wangpipe_colshuf" \
  "${COB}/results/RBBP4_wangpipe_colshuf" \
  "our-colShuf"

echo "[$(date)] done"
echo "lab perm para:"; cat "${LAB_PERM_PARA}"
echo "lab colShuf para:"; cat "${LAB_COL_PARA}"
