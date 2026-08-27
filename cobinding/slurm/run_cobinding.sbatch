#!/bin/bash
#SBATCH --job-name=cobinding
#SBATCH --output=cobinding-%j.out
#SBATCH --error=cobinding-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Higher-order co-binding clusters for one TF, from its Cicero connection files.
#   1. build_peak_graph.py : data/TF.<TF>.fitConns.res.{sel,pdc} -> peak graph
#   2. cluster_peaks.py    : peak graph -> communities + cliques + annotation
#
#   sbatch cobinding/slurm/run_cobinding.sbatch                 # TF=RBBP4
#   TF=RBBP4 QVAL=0.01 sbatch cobinding/slurm/run_cobinding.sbatch
#   REBUILD=0 QVAL=0.01 sbatch cobinding/slurm/run_cobinding.sbatch   # re-cluster only
#
# Full network (pp+dd+pd, the .sel file) makes the graph non-bipartite so true
# a-b-c cliques form; .pdc alone is proximal-distal (bipartite -> no triangles).

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
DATA_DIR="${DATA_DIR:-${XULAB}/data}"
WORK="${COB_WORK:-${COB}/work}"
RESULTS="${COB_RESULTS:-${COB}/results}"

TF="${TF:-RBBP4}"
# Input files for this TF: .sel (full pp+dd+pd) + .pdc (annotated). Override to
# restrict, e.g. GLOB='TF.{tf}.fitConns.res.sel' for full-network only.
# Do not put '{tf}' inside ${GLOB:-...} — bash treats the inner '}' as the end
# of the parameter expansion and the glob becomes TF.{tf.fitConns.res.*}
if [[ -z "${GLOB:-}" ]]; then
  GLOB='TF.{tf}.fitConns.res.*'
fi
REBUILD="${REBUILD:-1}"
Q_KEEP="${Q_KEEP:-1.0}"
QVAL="${QVAL:-0.05}"
COACCESS_MIN="${COACCESS_MIN:-0.0}"
MIN_SUPPORT="${MIN_SUPPORT:-1}"
METHOD="${METHOD:-louvain}"
MIN_CLIQUE="${MIN_CLIQUE:-3}"
MIN_CLUSTER="${MIN_CLUSTER:-3}"

CONDA_ENV="${CONDA_ENV:-cobinding}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${WORK}" "${RESULTS}/${TF}"
EDGES="${WORK}/${TF}.peak_edges.tsv"

if [[ "${REBUILD}" == "1" || ! -s "${EDGES}" ]]; then
  echo "[$(date)] build_peak_graph TF=${TF}"
  python "${COB}/scripts/build_peak_graph.py" \
    --data-dir "${DATA_DIR}" --tf "${TF}" --glob "${GLOB}" \
    --out-dir "${WORK}" --q-keep "${Q_KEEP}"
else
  echo "[$(date)] reuse ${EDGES}"
fi

echo "[$(date)] cluster_peaks TF=${TF} (q<${QVAL} coaccess>=${COACCESS_MIN} support>=${MIN_SUPPORT})"
python "${COB}/scripts/cluster_peaks.py" \
  --edges "${EDGES}" --out-dir "${RESULTS}/${TF}" \
  --qval "${QVAL}" --coaccess-min "${COACCESS_MIN}" --min-support "${MIN_SUPPORT}" \
  --method "${METHOD}" --min-clique "${MIN_CLIQUE}" --min-cluster "${MIN_CLUSTER}" --plot
echo "[$(date)] done -> ${RESULTS}/${TF}"
