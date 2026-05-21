#!/bin/bash
#SBATCH --job-name=scb_cis_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# Single-step pipeline. Run scBasset and standalone cisTopic first; their
# <work>/impute/ outputs are the inputs to this denoiser.

exec "$(dirname "${BASH_SOURCE[0]}")/01_denoise.sbatch" "$@"
