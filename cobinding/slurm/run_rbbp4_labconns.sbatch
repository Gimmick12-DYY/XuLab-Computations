#!/bin/bash
#SBATCH --job-name=rbbp4_labconns
#SBATCH --output=rbbp4_labconns-%j.out
#SBATCH --error=rbbp4_labconns-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#
# His downstream only: lab RBBP4.perm.fitConns.res.txt (coaccess + shuffle p/FDR)
# -> unique pairs -> cluster_peaks q<=0.05, coaccess>=0.
# Does NOT touch gold RBBP4 or /vast/som/xujie_lab/.
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
TAG="${TAG:-RBBP4_labconns}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"
CHROMHMM="${CHROMHMM:-${DATA_DIR}/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${COB}/work/RBBP4.peak_edges.tsv"
GOLD_SUMMARY="${COB}/results/RBBP4/summary.txt"
SEL="${DATA_DIR}/TF.RBBP4.fitConns.res.sel"
PERM_RES="${PERM_RES:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.res.txt}"
PERM_PARA="${PERM_PARA:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt}"

PY_ENV="${PY_ENV:-${XULAB}/../conda/envs/cobinding}"
[[ -x "${PY_ENV}/bin/python" ]] || PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PY_ENV}"

mkdir -p "${WORK}" "${RESULTS}"
echo "[$(date)] lab conns -> edges (precomputed perm p/FDR)  work=${WORK}"
[[ -s "${PERM_RES}" ]] || { echo "missing ${PERM_RES}" >&2; exit 1; }

# .sel-like: keep p<=0.05 (his published pair table), use his FDR.
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${PERM_RES}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method precomputed \
  --sel "${SEL}"

python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --plot --net-min-comp 3 --net-big 10 --max-dist 1000000

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
  --gold-summary "${GOLD_SUMMARY}" --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0

# Also: recompute p from his perm mu/sd on the same table (should match precomputed p).
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${PERM_RES}" --out-dir "${WORK}/from_para" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method shuffle --shuffle-para "${PERM_PARA}" \
  --sel "${SEL}"

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK}/from_para/RBBP4.peak_edges.tsv" \
  --out "${RESULTS}/compare_to_gold_from_para.txt" --qval 0.05 --coaccess-min 0

echo "[$(date)] done -> ${RESULTS}"
echo "==== precomputed FDR ===="
cat "${RESULTS}/compare_to_gold.txt"
echo "==== gaussian from para.txt ===="
cat "${RESULTS}/compare_to_gold_from_para.txt"
echo "para.txt: ${PERM_PARA}"
cat "${PERM_PARA}"
