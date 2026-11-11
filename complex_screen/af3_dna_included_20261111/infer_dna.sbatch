#!/bin/bash
#SBATCH -J af3dnainc
#SBATCH -p l40-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_dna_included_20261111/logs/%x_%A_%a.out
# DNA-included AF3 test (DECLARATION.md). Same AF3 3.0.1 container, flags and resources as af3_corr060_20261104/infer.sbatch; prebuilt JSONs (build_inputs.py), seeds 1-5.
# usage: sbatch --array=0-9%2 infer_dna.sbatch
A=/nas/longleaf/containers/alphafold/3.0.1; O=/work/users/d/y/dyy12/XuLab/complex_screen/af3_dna_included_20261111
module load singularity
mkdir -p $O/tmp $O/jaxcache $O/out; export TMPDIR=$O/tmp SINGULARITY_TMPDIR=$O/tmp APPTAINER_TMPDIR=$O/tmp
n=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/pairs.txt); l=${n,,}_dna; echo $n $(hostname); nvidia-smi -L
[ -s $O/out/$l/ranking_scores.csv ] && { echo already done; exit 0; }
ENVS=(--env XLA_PYTHON_CLIENT_PREALLOCATE=false --env TF_FORCE_UNIFIED_MEMORY=true --env XLA_CLIENT_MEM_FRACTION=3.2)
singularity exec --no-home --nv "${ENVS[@]}" --bind $O:/root/af --bind $A/weights:/root/models --bind /datacommons/alphafold/db_3.0.1:/root/public_databases \
  --bind $A/code:/root/code $A/image/alphafold3.0.1-cuda12.6-ubuntu22.04.sif python /root/code/alphafold3/run_alphafold.py --model_dir=/root/models --db_dir=/root/public_databases \
  --json_path=/root/af/run_in/$n.json --output_dir=/root/af/out --norun_data_pipeline --jax_compilation_cache_dir=/root/af/jaxcache; rc=$?
echo rc $rc; rm -f $O/out/$l/${l}_data.json; exit $rc
