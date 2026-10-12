#!/bin/bash
#SBATCH --job-name=rbbp4_w2mb
#SBATCH --output=rbbp4_w2mb-%j.out
#SBATCH --error=rbbp4_w2mb-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#
# Second Cicero search at a 2 Mb window, same k=50 / cap 10,000 / seed 555
# as the 1 Mb comparison run. Sidecar only. Gold files stay read-only.
# After scoring, keep pairs that are selected (p<=0.05) in BOTH windows.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
CICERO_ROOT="${COB}/cicero"
SRC="${COB}/work/RBBP4_labpmat/cicero"
ONE_MB_EDGES="${COB}/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv"
PERM_PARA="/vast/som/xujie_lab/TAGATG-293T/combined/Cell1000/cicero/RBBP4.perm.fitConns.para.txt"
SEL="${XULAB}/data/TF.RBBP4.fitConns.res.sel"
CHROMHMM="${XULAB}/data/HEK293T_chromHMM18.bed.gz"
PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
CICERO_ENV="${CICERO_ENV:-cicero}"

TAG="RBBP4_sweep_k50_i10000_s555_w2mb"
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
  k_metacell: 50
  metacell_max_iter: 10000
  window_bp: 2000000
  sample_num: 100
  max_cells: 0
  random_seed: 555
  shuffle: none
EOF

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CICERO_ENV}"
module unload r 2>/dev/null || true
cd "${CICERO_ROOT}"
Rscript scripts/03_run_cicero.R --config "${CFG}" --work-dir "${WORK}/cicero"

CONNS="${WORK}/cicero/cicero/coaccess.tsv.gz"

conda activate "${PY_ENV}"
cd "${XULAB}"
python "${COB}/scripts/cicero_conns_to_edges.py" \
  --tf RBBP4 --conns "${CONNS}" --out-dir "${WORK}" \
  --chromhmm "${CHROMHMM}" --max-dist 2000000 \
  --coaccess-min -1 --max-p 0.05 \
  --pval-method shuffle --shuffle-para "${PERM_PARA}" \
  --sel "${SEL}"

python - <<'PY'
from pathlib import Path
one = Path("/work/users/d/y/dyy12/XuLab/cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv")
two = Path("/work/users/d/y/dyy12/XuLab/cobinding/work/RBBP4_sweep_k50_i10000_s555_w2mb/RBBP4.peak_edges.tsv")
out = Path("/work/users/d/y/dyy12/XuLab/cobinding/work/RBBP4_sweep_k50_i10000_s555_w2mb/RBBP4.peak_edges.both_windows.tsv")

def key(a, b):
    return tuple(sorted((a, b)))

two_keys = set()
with two.open() as f:
    header = f.readline()
    cols = header.rstrip("\n").split("\t")
    i1, i2 = cols.index("peak1"), cols.index("peak2")
    for line in f:
        p = line.rstrip("\n").split("\t")
        two_keys.add(key(p[i1], p[i2]))

n_one = n_keep = 0
with one.open() as f, out.open("w") as o:
    header = f.readline()
    o.write(header)
    cols = header.rstrip("\n").split("\t")
    i1, i2 = cols.index("peak1"), cols.index("peak2")
    for line in f:
        n_one += 1
        p = line.rstrip("\n").split("\t")
        if key(p[i1], p[i2]) in two_keys:
            o.write(line if line.endswith("\n") else line + "\n")
            n_keep += 1
print(f"1Mb selected {n_one}")
print(f"2Mb selected {len(two_keys)}")
print(f"in both {n_keep}")
print(f"wrote {out}")
PY

python "${COB}/scripts/cluster_peaks.py" \
  --edges "${WORK}/RBBP4.peak_edges.both_windows.tsv" --out-dir "${RESULTS}" \
  --qval 0.05 --coaccess-min 0 --min-support 1 \
  --min-clique 3 --share 2 --net-min-comp 3 --net-big 10 \
  --max-dist 2000000

echo "DONE ${TAG}"
cat "${RESULTS}/summary.txt"
