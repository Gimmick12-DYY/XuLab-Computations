# shared AF3 container settings (from the module's example job); caches and temp in the run folder, no home access
A=/nas/longleaf/containers/alphafold/3.0.1; O=/work/users/d/y/dyy12/XuLab/complex_screen/af3_novel_20261103
module load singularity
export TMPDIR=$O/tmp SINGULARITY_TMPDIR=$O/tmp APPTAINER_TMPDIR=$O/tmp
run_af3() { singularity exec --no-home --nv "${ENVS[@]}" --bind $O:/root/af --bind $A/weights:/root/models --bind /datacommons/alphafold/db_3.0.1:/root/public_databases \
  --bind $A/code:/root/code $A/image/alphafold3.0.1-cuda12.6-ubuntu22.04.sif python /root/code/alphafold3/run_alphafold.py --model_dir=/root/models --db_dir=/root/public_databases "$@"; }
