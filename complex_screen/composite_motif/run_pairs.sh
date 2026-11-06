O=/work/users/d/y/dyy12/XuLab/complex_screen/composite_motif; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
F=$(sbatch --parsable -J cmfimo -p general -c 1 --mem=16G -t 03:00:00 -o $O/logs/fimo_%j.out --wrap "cd $O; $PY cm_fimo.py")
P=$(sbatch --parsable --dependency=afterok:$F -J cmpair -p general -c 1 --mem=16G -t 04:00:00 -o $O/logs/pairs_%j.out --wrap "cd $O; $PY cm_pairs.py")
echo fimo $F pairs $P
