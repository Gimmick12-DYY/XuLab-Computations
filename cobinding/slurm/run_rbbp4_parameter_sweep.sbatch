#!/bin/bash
#SBATCH --job-name=rbbp4_sweep
#SBATCH --output=rbbp4_sweep-%A_%a.out
#SBATCH --error=rbbp4_sweep-%A_%a.err
#SBATCH --array=0-7%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# Sweep Cicero aggregation parameters on the already-built, exact RBBP4 peak
# matrix/CDS. Gold files under /vast and cobinding/work/RBBP4* are read-only.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${COB}/cicero"
SRC="${COB}/work/RBBP4_labpmat/cicero"
GOLD_EDGES="${COB}/work/RBBP4.peak_edges.tsv"
GOLD_SUMMARY="${COB}/results/RBBP4/summary.txt"
LAB_CONNS="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4/RBBP4.conns.rds"
PERM_PARA="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt"
SEL="${XULAB}/data/TF.RBBP4.fitConns.res.sel"
CHROMHMM="${XULAB}/data/HEK293T_chromHMM18.bed.gz"
PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

# k, metacell iteration cap, seed. The prior best region was k=50:
# cap 5k -> 11,230 selected; cap 20k -> 20,100 selected.
PARAMS=(
  "40 10000 555"
  "50 8000 555"
  "50 10000 555"
  "50 12000 555"
  "50 15000 555"
  "60 10000 555"
  "75 10000 555"
  "50 10000 2020"
)
read -r K MAX_ITER SEED <<< "${PARAMS[${SLURM_ARRAY_TASK_ID}]}"

TAG="RBBP4_sweep_k${K}_i${MAX_ITER}_s${SEED}"
WORK="${COB}/work/${TAG}"
RESULTS="${COB}/results/${TAG}"
CFG="${WORK}/config.yaml"
mkdir -p "${WORK}/cicero" "${RESULTS}" "${WORK}/tmp"
ln -sfn "${SRC}/mm" "${WORK}/cicero/mm"
ln -sfn "${SRC}/cds" "${WORK}/cicero/cds"
export TMPDIR="${WORK}/tmp"

cat > "${CFG}" <<EOF
paths:
  work_dir: ${WORK}/cicero
cicero:
  min_cells_per_peak: 1
  num_dim: 50
  k_metacell: ${K}
  metacell_max_iter: ${MAX_ITER}
  window_bp: 1000000
  sample_num: 100
  max_cells: 0
  random_seed: ${SEED}
  shuffle: none
EOF

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"
Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${WORK}/cicero"

CONNS="${WORK}/cicero/cicero/coaccess.tsv.gz"
Rscript "${COB}/scripts/compare_raw_cicero_conns.R" \
  --lab-conns "${LAB_CONNS}" --ours "${CONNS}" \
  --out "${RESULTS}/compare_to_lab_conns.txt"

conda activate "${PY_ENV}"
cd "${XULAB}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 1000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method shuffle --shuffle-para "${PERM_PARA}" \
  --sel "${SEL}"

python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.tsv" --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --net-min-comp 3 --net-big 10 \
  --max-dist 1000000

python "${COB}/scripts/compare_rbbp4_repro.py" \
  --gold-edges "${GOLD_EDGES}" --repro-edges "${WORK}/RBBP4.peak_edges.tsv" \
  --gold-summary "${GOLD_SUMMARY}" --repro-summary "${RESULTS}/summary.txt" \
  --out "${RESULTS}/compare_to_gold.txt" --qval 0.05 --coaccess-min 0

cat "${WORK}/cicero/cicero/cicero_info.tsv"
cat "${RESULTS}/compare_to_gold.txt"
