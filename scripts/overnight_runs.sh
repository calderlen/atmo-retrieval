#!/usr/bin/env bash
# ==============================================================================
# Overnight MCMC Run Script
# ==============================================================================
# Runs multiple retrieval attempts with progressively more species.
# Robust to GPU OOM errors - continues to next run if one fails.
#
# Usage:
#   nohup bash scripts/overnight_runs.sh &
#   # or in tmux:
#   tmux new -s overnight
#   bash scripts/overnight_runs.sh
#   # Ctrl+B, D to detach
#
# Customize RUN_CONFIGS below to change species/params for each run.
# ==============================================================================

set -uo pipefail  # Don't use -e, we want to continue on failure

# Required arguments (no defaults)
PLANET="${PLANET:-}"
EPOCH="${EPOCH:-}"
MODE="${MODE:-transmission}"

# Parse command line
while [[ $# -gt 0 ]]; do
  case "$1" in
    --planet) PLANET="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 --planet PLANET --epoch EPOCH [--mode MODE]"
      echo ""
      echo "Required:"
      echo "  --planet    Planet name (e.g., KELT-20b)"
      echo "  --epoch     Observation epoch YYYYMMDD"
      echo ""
      echo "Optional:"
      echo "  --mode      Retrieval mode (default: transmission)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$PLANET" ]] || [[ -z "$EPOCH" ]]; then
  echo "Error: --planet and --epoch are required"
  echo "Usage: $0 --planet PLANET --epoch EPOCH [--mode MODE]"
  exit 1
fi

# ==============================================================================
# RUN CONFIGURATIONS
# ==============================================================================
# Each run is defined as: "name|atoms|molecules|extra_args"
# Runs are ordered from simplest (most likely to succeed) to most complex.
# The script will try each run in order; OOM or other errors won't stop later runs.

RUN_CONFIGS=(
  # Run 1: Minimal - just iron (fastest, most likely to complete)
  "minimal|Fe I,Fe II||--mcmc-warmup 1000 --mcmc-samples 1000"
  
  # Run 2: Core metals - iron + sodium (detected in most UHJ studies)  
  "core_metals|Fe I,Fe II,Na I||--mcmc-warmup 1500 --mcmc-samples 1500"
  
  # Run 3: Extended atoms - add calcium and chromium
  "extended_atoms|Fe I,Fe II,Na I,Ca II,Cr I,Cr II||--mcmc-warmup 2000 --mcmc-samples 2000"
  
  # Run 4: Full default species (7 atoms + FeH molecule)
  "full_default|Na I,Mg I,Ca II,Cr I,Cr II,Fe I,Fe II|FeH|--mcmc-warmup 2000 --mcmc-samples 2000"
  
  # Run 5: Full default + extra molecules (most ambitious)
  "full_molecules|Na I,Mg I,Ca II,Cr I,Cr II,Fe I,Fe II|FeH,H2O,CO|--mcmc-warmup 2000 --mcmc-samples 2000"
)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/overnight_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

STATUS_LOG="$LOG_DIR/status.log"
echo "Overnight run started at $(date)" | tee "$STATUS_LOG"
echo "Planet: $PLANET, Epoch: $EPOCH, Mode: $MODE" | tee -a "$STATUS_LOG"
echo "Log directory: $LOG_DIR" | tee -a "$STATUS_LOG"
echo "========================================" | tee -a "$STATUS_LOG"

# Track results
declare -A RESULTS

# ==============================================================================
# OOM DETECTION AND ERROR HANDLING
# ==============================================================================

is_oom_error() {
  local log_file="$1"
  # Check for common GPU OOM patterns
  grep -qE "(out of memory|OOM|CUDA error|Resource exhausted|XLA.*memory)" "$log_file" 2>/dev/null
}

# ==============================================================================
# MAIN LOOP
# ==============================================================================

for run_config in "${RUN_CONFIGS[@]}"; do
  IFS='|' read -r run_name atoms molecules extra_args <<< "$run_config"
  
  echo "" | tee -a "$STATUS_LOG"
  echo "========================================" | tee -a "$STATUS_LOG"
  echo "Starting run: $run_name" | tee -a "$STATUS_LOG"
  echo "Atoms: ${atoms:-none}" | tee -a "$STATUS_LOG"
  echo "Molecules: ${molecules:-none}" | tee -a "$STATUS_LOG"
  echo "Started at: $(date)" | tee -a "$STATUS_LOG"
  echo "========================================" | tee -a "$STATUS_LOG"
  
  RUN_LOG="$LOG_DIR/${run_name}.log"
  
  # Build command (run from project root)
  cmd=(python __main__.py 
       --planet "$PLANET" 
       --epoch "$EPOCH"
       --mode "$MODE"
       --load-opacities
       --no-preallocate)
  
  # Add species if specified
  if [[ -n "$atoms" ]]; then
    cmd+=(--atoms "$atoms")
  fi
  if [[ -n "$molecules" ]]; then
    cmd+=(--molecules "$molecules")
  fi
  
  # For atoms-only runs
  if [[ -z "$molecules" ]]; then
    cmd+=(--no-molecules)
  fi
  
  # Add extra args
  if [[ -n "$extra_args" ]]; then
    # shellcheck disable=SC2206
    cmd+=($extra_args)
  fi
  
  echo "Command: ${cmd[*]}" | tee -a "$STATUS_LOG"
  
  # Run with timeout (8 hours max per run)
  start_time=$(date +%s)
  
  # Run and capture exit code (don't exit on failure due to set +e behavior)
  set +e
  timeout 8h "${cmd[@]}" 2>&1 | tee "$RUN_LOG"
  exit_code=${PIPESTATUS[0]}
  set -e
  
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  duration_hrs=$(echo "scale=2; $duration / 3600" | bc)
  
  # Analyze result
  if [[ $exit_code -eq 0 ]]; then
    RESULTS[$run_name]="SUCCESS"
    echo "✅ $run_name: SUCCESS (${duration_hrs}h)" | tee -a "$STATUS_LOG"
  elif [[ $exit_code -eq 124 ]]; then
    RESULTS[$run_name]="TIMEOUT"
    echo "⏱️  $run_name: TIMEOUT after 8h" | tee -a "$STATUS_LOG"
  elif is_oom_error "$RUN_LOG"; then
    RESULTS[$run_name]="OOM"
    echo "💥 $run_name: GPU OOM ERROR (${duration_hrs}h)" | tee -a "$STATUS_LOG"
    echo "   Continuing to next run..." | tee -a "$STATUS_LOG"
  else
    RESULTS[$run_name]="FAILED"
    echo "❌ $run_name: FAILED with exit code $exit_code (${duration_hrs}h)" | tee -a "$STATUS_LOG"
  fi
  
  # Small pause to let GPU memory fully clear
  sleep 10
done

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "" | tee -a "$STATUS_LOG"
echo "========================================" | tee -a "$STATUS_LOG"
echo "OVERNIGHT RUN SUMMARY" | tee -a "$STATUS_LOG"
echo "Completed at: $(date)" | tee -a "$STATUS_LOG"
echo "========================================" | tee -a "$STATUS_LOG"

successes=0
failures=0

for run_name in "${!RESULTS[@]}"; do
  result="${RESULTS[$run_name]}"
  echo "$run_name: $result" | tee -a "$STATUS_LOG"
  if [[ "$result" == "SUCCESS" ]]; then
    ((successes++))
  else
    ((failures++))
  fi
done

echo "" | tee -a "$STATUS_LOG"
echo "Total: $successes succeeded, $failures failed" | tee -a "$STATUS_LOG"
echo "Logs saved to: $LOG_DIR" | tee -a "$STATUS_LOG"

# Exit with 0 if at least one run succeeded
if [[ $successes -gt 0 ]]; then
  exit 0
else
  exit 1
fi
