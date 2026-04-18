#!/bin/bash
#SBATCH --account=PAS2489
#SBATCH --job-name=k20b_20190504
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster ascend
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
trap 'echo "[$(date)] Failed on line $LINENO"' ERR

cd "$SLURM_SUBMIT_DIR"

echo "[$(date)] Starting ${SLURM_JOB_NAME:-local_job} in ${PWD}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

#export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
#export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

nvidia-smi -L || true

# TODO: replace with your OSC environment Python executable.
PYTHON_BIN="/path/to/osc/envs/retrieval/bin/python"
BANDPASS_TBL="input/phot/transmission/kelt20b/kelt20b_tess_bandpass.tbl"
FASTCHEM_PARAMETER_FILE="input/fastchem/parameters.dat"

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Bandpass table: $BANDPASS_TBL"
echo "FastChem parameter file: $FASTCHEM_PARAMETER_FILE"
echo "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"

/usr/bin/time -v srun --ntasks=1 "$PYTHON_BIN" -m atmo_retrieval \
  --profile hpc \
  --planet KELT-20b \
  --mode transmission \
  --epoch 20190504 \
  --bandpass-tbl "$BANDPASS_TBL" \
  --data-format timeseries \
  --wavelength-range full \
  --chemistry-model fastchem_hybrid_grid \
  --atoms "Fe I,Ni I,Cr I,Na I" \
  --no-molecules \
  --pt-profile guillot \
  --fastchem-parameter-file "$FASTCHEM_PARAMETER_FILE" \
  --load-opacities \
  --resolution-mode hr \
  --nlayer 20 \
  --n-spectral-points 130000 \
  --mcmc-chains 2 \
  --mcmc-chain-method sequential \
  --mcmc-warmup 2000 \
  --mcmc-samples 2000 \
  --svi-steps 8000 \
  --svi-learning-rate 0.001 \
  --svi-lr-decay-steps 2000 \
  --svi-lr-decay-rate 0.5

echo "[$(date)] Job finished"
