#!/bin/bash
#SBATCH -J pilot_panels
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/logs/panels_%j.out
# layer 2 scores -> PyMOL renders -> panel SVGs
set -euo pipefail
P=/work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs
/work/users/d/y/dyy12/envs/boltz/bin/python -I $P/pilot_layer2.py
/work/users/d/y/dyy12/envs/pymol3d/bin/pymol -cq $P/pilot_render.py -- $P
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I $P/pilot_panels.py 2>&1 | grep -v findfont
ls $P/panels
