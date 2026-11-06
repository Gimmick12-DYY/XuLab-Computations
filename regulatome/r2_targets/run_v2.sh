O=/work/users/d/y/dyy12/XuLab/regulatome/r2_targets; PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
A=$(sbatch --parsable -J b0arms -p general -c 1 --mem=16G -t 02:00:00 -o $O/logs/b0_%j.out --wrap "$PY -I $O/b0_arms.py")
C=$(sbatch --parsable -J cisv2 -p general -c 1 --mem=16G -t 02:00:00 -o $O/logs/cisv2_%j.out --wrap "$PY -I $O/r2_cis_v2.py")
echo arms $A cis $C
