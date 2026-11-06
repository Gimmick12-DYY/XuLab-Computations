#!/bin/bash
#SBATCH -p l40-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/logs/gpu_test_%j.out
nvidia-smi | head -12
source /work/users/d/y/dyy12/envs/boltz/bin/activate
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl -s -m 20 -o /dev/null -w "colabfold http %{http_code}\n" https://api.colabfold.com/queue || echo "colabfold unreachable"
