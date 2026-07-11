#!/bin/bash
#SBATCH --account=PAS2489
#SBATCH --job-name=k20b_diag
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster ascend
#SBATCH --partition=gpu
#SBATCH --array=0-27
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
trap 'echo "[$(date)] Failed on line $LINENO"' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

CASES=(
  tiny_red_fe_only
  tiny_red_fe_only_no_svi
  tiny_red_fe_only_sigma1p5
  tiny_red_fe_only_sigma2
  tiny_red_fe_only_sigma3
  tiny_red_fe_only_sigma5
  tiny_red_fe_only_sigma8
  tiny_red_fe_only_x64
  tiny_red_fe_only_no_sysrem
  tiny_red_fe_only_stride2
  tiny_red_fe_only_stride4
  tiny_red_fe_only_stride8
  red_all_atoms
  tiny_blue_fe_only
  tiny_blue_fe_only_no_svi
  tiny_blue_fe_only_sigma1p5
  tiny_blue_fe_only_sigma2
  tiny_blue_fe_only_sigma3
  tiny_blue_fe_only_sigma5
  tiny_blue_fe_only_sigma8
  tiny_blue_fe_only_x64
  tiny_blue_fe_only_no_sysrem
  tiny_blue_fe_only_stride2
  tiny_blue_fe_only_stride4
  tiny_blue_fe_only_stride8
  blue_all_atoms
  full_arm_all_atoms
  full_arm_no_sysrem
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#CASES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; valid range is 0-$(( ${#CASES[@]} - 1 ))" >&2
  exit 2
fi

CASE_LABEL="${CASES[$TASK_ID]}"

echo "[$(date)] Starting ${SLURM_JOB_NAME:-local_job} case=${CASE_LABEL} in ${PWD}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-local}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# Keep x64 controlled by the individual matrix case; JAX reads this at import time.
export JAX_ENABLE_X64=0

nvidia-smi -L || true

# Cluster/local knobs. Override these with environment variables if needed.
PYTHON_BIN="${PYTHON_BIN:-python}"
BANDPASS_TBL="${BANDPASS_TBL:-input/phot/transmission/kelt20b/kelt20b_tess_bandpass.tbl}"
FASTCHEM_PARAMETER_FILE="${FASTCHEM_PARAMETER_FILE:-input/fastchem/parameters.dat}"

COMMON_ARGS=(
  -m atmo_retrieval
  --profile hpc
  --planet KELT-20b
  --mode transmission
  --epoch 20190504
  --bandpass-tbl "$BANDPASS_TBL"
  --data-format timeseries
  --chemistry-model fastchem_hybrid_grid
  --pt-profile guillot
  --fastchem-parameter-file "$FASTCHEM_PARAMETER_FILE"
  --load-opacities
  --resolution-mode hr
  --mcmc-warmup 200
  --mcmc-samples 200
  --mcmc-chains 2
  --mcmc-chain-method sequential
  --svi-steps 2000
  --svi-learning-rate 0.001
  --svi-lr-decay-steps 2000
  --svi-lr-decay-rate 0.5
  --save-mcmc-diagnostics
  --diagnostic-label "$CASE_LABEL"
)

CASE_ARGS=()
case "$CASE_LABEL" in
  tiny_red_fe_only)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000)
    ;;
  tiny_red_fe_only_no_svi)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --skip-svi)
    ;;
  tiny_red_fe_only_sigma1p5)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 1.5)
    ;;
  tiny_red_fe_only_sigma2)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 2)
    ;;
  tiny_red_fe_only_sigma3)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 3)
    ;;
  tiny_red_fe_only_sigma5)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 5)
    ;;
  tiny_red_fe_only_sigma8)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 8)
    ;;
  tiny_red_fe_only_x64)
    export JAX_ENABLE_X64=1
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000)
    ;;
  tiny_red_fe_only_no_sysrem)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --no-sysrem)
    ;;
  tiny_red_fe_only_stride2)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 2)
    ;;
  tiny_red_fe_only_stride4)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 4)
    ;;
  tiny_red_fe_only_stride8)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 8)
    ;;
  red_all_atoms)
    CASE_ARGS=(--wavelength-range red --atoms "Fe I,Ni I,Cr I,Na I" --no-molecules --nlayer 20 --n-spectral-points 50000)
    ;;
  tiny_blue_fe_only)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000)
    ;;
  tiny_blue_fe_only_no_svi)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --skip-svi)
    ;;
  tiny_blue_fe_only_sigma1p5)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 1.5)
    ;;
  tiny_blue_fe_only_sigma2)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 2)
    ;;
  tiny_blue_fe_only_sigma3)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 3)
    ;;
  tiny_blue_fe_only_sigma5)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 5)
    ;;
  tiny_blue_fe_only_sigma8)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --sigma-scale 8)
    ;;
  tiny_blue_fe_only_x64)
    export JAX_ENABLE_X64=1
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000)
    ;;
  tiny_blue_fe_only_no_sysrem)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --no-sysrem)
    ;;
  tiny_blue_fe_only_stride2)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 2)
    ;;
  tiny_blue_fe_only_stride4)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 4)
    ;;
  tiny_blue_fe_only_stride8)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I" --no-molecules --nlayer 10 --n-spectral-points 30000 --spectral-stride 8)
    ;;
  blue_all_atoms)
    CASE_ARGS=(--wavelength-range blue --atoms "Fe I,Ni I,Cr I,Na I" --no-molecules --nlayer 20 --n-spectral-points 50000)
    ;;
  full_arm_all_atoms)
    CASE_ARGS=(--wavelength-range full --atoms "Fe I,Ni I,Cr I,Na I" --no-molecules --nlayer 20 --n-spectral-points 70000)
    ;;
  full_arm_no_sysrem)
    CASE_ARGS=(--wavelength-range full --atoms "Fe I,Ni I,Cr I,Na I" --no-molecules --nlayer 20 --n-spectral-points 70000 --no-sysrem)
    ;;
  *)
    echo "Unhandled case: $CASE_LABEL" >&2
    exit 2
    ;;
esac

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Bandpass table: $BANDPASS_TBL"
echo "FastChem parameter file: $FASTCHEM_PARAMETER_FILE"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "Command:"
printf '  %q' "$PYTHON_BIN" "${COMMON_ARGS[@]}" "${CASE_ARGS[@]}"
printf '\n'

RUN_PREFIX=()
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  RUN_PREFIX=(srun --ntasks=1)
fi

/usr/bin/time -v "${RUN_PREFIX[@]}" "$PYTHON_BIN" "${COMMON_ARGS[@]}" "${CASE_ARGS[@]}"

echo "[$(date)] Case ${CASE_LABEL} finished"
