#!/bin/bash
#SBATCH --job-name=k20b_bonidie_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

# ---- User-tunable paths ----------------------------------------------------
ATMO_REPO="${ATMO_REPO:-/home/calder/code/atmo-retrieval}"
CONDA_SH="${CONDA_SH:-/home/calder/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-retrieval}"
FASTCHEM_PARAM_FILE="${FASTCHEM_PARAM_FILE:-input/fastchem/parameters_py.dat}"
FASTCHEM_LOGK_FILE="${FASTCHEM_LOGK_FILE:-input/FastChem/input/logK/logK.dat}"

cd "${ATMO_REPO}"
mkdir -p logs

echo "[$(date)] Starting ${SLURM_JOB_NAME:-local_job} in ${PWD}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm}"

if [[ ! -f "${FASTCHEM_PARAM_FILE}" ]]; then
    echo "Missing FastChem parameter file: ${FASTCHEM_PARAM_FILE}" >&2
    exit 1
fi

if [[ ! -f "${FASTCHEM_LOGK_FILE}" ]]; then
    echo "Missing FastChem gas-species file: ${FASTCHEM_LOGK_FILE}" >&2
    echo "Place a FastChem checkout or copied FastChem input bundle under input/FastChem/," >&2
    echo "or override FASTCHEM_LOGK_FILE when submitting." >&2
    exit 1
fi

if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    echo "Set CONDA_SH=/path/to/conda.sh when submitting." >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Python: $(command -v python)"
python --version

python -m atmo_retrieval \
  --config-profile hpc \
  --planet KELT-20b \
  --mode transmission \
  --epoch 20190504 \
  --data-format timeseries \
  --wavelength-range full \
  --chemistry-model bonidie \
  --pt-profile guillot \
  --fastchem-parameter-file "${FASTCHEM_PARAM_FILE}" \
  --no-molecules \
  --atoms "Fe I" \
  --load-opacities \
  --resolution-mode hr \
  --svi-steps 10000 \
  --svi-learning-rate 0.001 \
  --svi-lr-decay-steps 2000 \
  --svi-lr-decay-rate 0.5 \
  --no-preallocate

echo "[$(date)] Job finished"
