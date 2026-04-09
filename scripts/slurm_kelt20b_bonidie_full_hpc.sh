#!/bin/bash
#SBATCH --job-name=retrieval-260409-1-kelt20b-20190504-bonidie-guillot-full
#SBATCH --chdir=/home/calder/code/atmo-retrieval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu
#SBATCH --account=PAS2489
#SBATCH --mail-type=ALL 

set -euo pipefail

echo "[$(date)] Starting ${SLURM_JOB_NAME:-local_job} in ${PWD}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm}"

if [[ ! -f input/fastchem/parameters_py.dat ]]; then
    echo "Missing FastChem parameter file: input/fastchem/parameters_py.dat" >&2
    exit 1
fi

if [[ -f /home/calder/miniforge3/etc/profile.d/conda.sh ]]; then
    source /home/calder/miniforge3/etc/profile.d/conda.sh
    conda activate retrieval
else
    echo "Conda activation script not found: /home/calder/miniforge3/etc/profile.d/conda.sh" >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Python: $(command -v python)"
python --version

srun --ntasks=1 python -m atmo_retrieval \
  --profile hpc \
  --planet KELT-20b \
  --mode transmission \
  --epoch 20190504 \
  --data-format timeseries \
  --wavelength-range full \
  --chemistry-model bonidie \
  --pt-profile guillot \
  --fastchem-parameter-file input/fastchem/parameters_py.dat \
  --load-opacities \
  --resolution-mode hr \
  --mcmc-chains 4 \
  --mcmc-chain-method parallel \
  --require-gpu-per-chain \
  --svi-steps 10000 \
  --svi-learning-rate 0.001 \
  --svi-lr-decay-steps 2000 \
  --svi-lr-decay-rate 0.5 \
  --no-preallocate

echo "[$(date)] Job finished"
