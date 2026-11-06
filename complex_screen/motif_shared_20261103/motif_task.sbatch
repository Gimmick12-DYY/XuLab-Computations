#!/bin/bash
#SBATCH -J motshare
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103/logs/task_%A_%a.out
O=/work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103; module load meme/5.5.7
t=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/tfs.txt); F=$O/fg/$t; echo $t $(hostname)
sea --p $F/fg.fa --n $F/bg.fa --m $O/db/db.meme --thresh 1000 --noseqs --oc $O/sea/$t; echo sea rc $?
streme --p $F/fg.fa --n $F/bg.fa --dna --minw 6 --maxw 16 --time 1500 --oc $O/streme/$t; echo streme rc $?
tomtom -dist pearson -min-overlap 5 -thresh 0.05 -no-ssc -oc $O/tomtom/$t $O/streme/$t/streme.txt $O/db/db.meme; echo tomtom rc $?
