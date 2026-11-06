X=/work/users/d/y/dyy12/XuLab; R=$X/regulatome/r1_tftf; T=$X/regulatome/r2_targets; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -I $R/r1_final_classes.py --freeze && $PY -I $T/r2_propagate.py && $PY -I $X/complex_screen/composite_motif/complex_counts_v3.py
