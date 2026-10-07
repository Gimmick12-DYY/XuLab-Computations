#!/bin/bash
#SBATCH --job-name=rbbp4_exact_fitconns
#SBATCH --output=rbbp4_exact_fitconns-%j.out
#SBATCH --error=rbbp4_exact_fitconns-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#
# Reconstruct Wang's RBBP4 fitConns directly from the raw Cicero conns RDS:
#   raw coaccess -> Wang perm Gaussian p -> BH FDR -> p<=0.05 selected pairs.
# This reproduces pair identities/scores without reading perm.fitConns.res.txt.
# It intentionally does not rerun Cicero from pmat: exact coaccess requires the
# original Cicero metacell/embedding realization, preserved in RBBP4.conns.rds.
#
# Gold and /vast/som/xujie_lab are read-only. Outputs are sidecars under
# cobinding/work|results/RBBP4_exact_fitconns.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
TAG="${TAG:-RBBP4_exact_fitconns}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"

LAB_CONNS_RDS="${LAB_CONNS_RDS:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds}"
PERM_PARA="${PERM_PARA:-/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt}"
SEL="${SEL:-${XULAB}/data/TF.RBBP4.fitConns.res.sel}"
CHROMHMM="${CHROMHMM:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"
GOLD_EDGES="${GOLD_EDGES:-${COB}/work/RBBP4.peak_edges.tsv}"
GOLD_SUMMARY="${GOLD_SUMMARY:-${COB}/results/RBBP4/summary.txt}"

PY_ENV="${PY_ENV:-/work/users/d/y/dyy12/conda/envs/cobinding}"
CICERO_ENV="${CICERO_ENV:-cicero}"

[[ -s "${LAB_CONNS_RDS}" ]] || { echo "missing ${LAB_CONNS_RDS}" >&2; exit 1; }
[[ -s "${PERM_PARA}" ]] || { echo "missing ${PERM_PARA}" >&2; exit 1; }
[[ -x "${PY_ENV}/bin/python" ]] || { echo "missing Python env ${PY_ENV}" >&2; exit 1; }

mkdir -p "${WORK}" "${RESULTS}" "${WORK}/tmp"
export TMPDIR="${WORK}/tmp"
source "$(conda info --base)/etc/profile.d/conda.sh"

FITCONNS="${WORK}/RBBP4.perm.fitConns.res.txt.gz"

echo "[$(date)] raw Wang Cicero conns -> complete fitConns table"
conda activate "${CICERO_ENV}"
Rscript "${COB}/scripts/score_cicero_rds_fitconns.R" \
  --conns-rds "${LAB_CONNS_RDS}" \
  --shuffle-para "${PERM_PARA}" \
  --output "${FITCONNS}"

echo "[$(date)] fitConns -> p<=0.05 selected pairs"
conda activate "${PY_ENV}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${FITCONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method precomputed \
  --sel "${SEL}"

echo "[$(date)] cluster FDR<=0.05 selected pairs"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" \
  --annot "${WORK}/RBBP4.peak_annot.tsv" \
  --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --plot \
  --net-min-comp 3 --net-big 10 --max-dist 1000000

if [[ -s "${GOLD_EDGES}" ]]; then
  python "${COB}/scripts/compare_rbbp4_repro.py" \
    --gold-edges "${GOLD_EDGES}" \
    --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
    --gold-summary "${GOLD_SUMMARY}" \
    --repro-summary "${RESULTS}/summary.txt" \
    --out "${RESULTS}/compare_to_gold.txt" \
    --qval 0.05 --coaccess-min 0
  cat "${RESULTS}/compare_to_gold.txt"
fi

echo "[$(date)] exact fitConns reconstruction complete"
echo "all scored pairs: ${FITCONNS}"
echo "p<=0.05 pairs:    ${WORK}/RBBP4.peak_edges.tsv"
echo "cliques/modules:  ${RESULTS}"
