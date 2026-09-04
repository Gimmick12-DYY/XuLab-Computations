#!/bin/bash
#SBATCH --job-name=tf_complexes
#SBATCH --output=tf_complexes-%j.out
#SBATCH --error=tf_complexes-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# Step 2: union per-TF peaks -> loci x TF matrix, remove HOT loci + blacklist,
# correlate TFs (pearson/jaccard), cluster -> candidate protein complexes
# (Partridge 2020 style). A/B compartments annotate the complexes (they do NOT
# define co-binding).
#
#   sbatch tf_complex/slurm/02_tf_complexes.sbatch
#   METRIC=jaccard HOT_FRAC=0.4 CLUSTER_THRESHOLD=0.5 sbatch tf_complex/slurm/02_tf_complexes.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
SOURCE="${SOURCE:-raw}"                          # raw | imputed (must match step 01)
SUF=""; [[ "${SOURCE}" == "imputed" ]] && SUF="_imputed"
PEAKS_DIR="${PEAKS_DIR:-${TFC}/work/peaks${SUF}}"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work${SUF}}"
RESULTS="${RESULTS:-${TFC}/results${SUF}}"
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
BLACKLIST="${BLACKLIST:-${TFC}/ref/hg38-blacklist.v2.bed}"   # optional; skipped if missing

METRIC="${METRIC:-pearson}"             # pearson (phi, default) | jaccard  (binary signal only)
SIGNAL="${SIGNAL:-cpm}"                 # cpm (CPM-normalized signalValue; de-confounds depth at the
                                        # input, default) | binary (peak presence 0/1)
HOT_FRAC="${HOT_FRAC:-0.5}"             # drop loci bound by > this fraction of TFs (HOT artifacts)
MIN_OCC="${MIN_OCC:-2}"                 # keep loci bound by >= this many TFs
CLUSTER="${CLUSTER:-hierarchical}"      # hierarchical | threshold
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-0.2}"   # pearson co-occupancy is small in abs terms; set near printed p95
SPQN="${SPQN:-0}"                       # 1 = Spatial Quantile Normalization (remove the cell-count/
                                        # complexity confound). NOTE SpQN compresses the scale ->
                                        # lower CLUSTER_THRESHOLD to the printed post-SpQN p95.
SPQN_NGROUP="${SPQN_NGROUP:-10}"; SPQN_W="${SPQN_W:-20}"; SPQN_REF_FRAC="${SPQN_REF_FRAC:-0.8}"
REBUILD="${REBUILD:-1}"                 # 0 = reuse peak_matrix.npz

CONDA_ENV="${CONDA_ENV:-tf_complex}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${AB_BED}" ]] || { echo "missing A/B calls: ${AB_BED}" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

# CPM needs signal_matrix.npz (only produced by the current builder) -> force a rebuild if absent
NEED_SIG=0; [[ "${SIGNAL}" == "cpm" && ! -s "${MATRIX_DIR}/signal_matrix.npz" ]] && NEED_SIG=1
if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/peak_matrix.npz" || "${NEED_SIG}" == "1" ]]; then
  echo "[$(date)] build_peak_matrix (union per-TF peaks -> loci x TF + signal_matrix)"
  python "${TFC}/scripts/build_peak_matrix.py" \
    --peaks-dir "${PEAKS_DIR}" --out-dir "${MATRIX_DIR}" --ab-bed "${AB_BED}"
else
  echo "[$(date)] reuse ${MATRIX_DIR}/peak_matrix.npz"
fi

BL_ARG=()
[[ -s "${BLACKLIST}" ]] && BL_ARG=(--blacklist "${BLACKLIST}") || echo "[warn] no blacklist at ${BLACKLIST}; skipping blacklist filter"

echo "[$(date)] tf_complexes (metric=${METRIC}, hot_frac=${HOT_FRAC})"
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"   # per-TF cell counts for the complexity-format heatmap
SPQN_ARG=(); [[ "${SPQN}" == "1" ]] && SPQN_ARG=(--spqn --spqn-ngroup "${SPQN_NGROUP}" --spqn-w "${SPQN_W}" --spqn-ref-frac "${SPQN_REF_FRAC}")
python "${TFC}/scripts/tf_complexes.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --metric "${METRIC}" --signal "${SIGNAL}" --cell-meta "${CELL_META}" \
  --hot-frac "${HOT_FRAC}" --min-occ "${MIN_OCC}" \
  --cluster "${CLUSTER}" --cluster-threshold "${CLUSTER_THRESHOLD}" \
  "${SPQN_ARG[@]}" "${BL_ARG[@]}" --plot
echo "[$(date)] done -> ${RESULTS}  (corr_complexity_{genome,A,B}.png = complexity format)"
