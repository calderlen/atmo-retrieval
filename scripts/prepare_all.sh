#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="input/raw"
ARMS="red"
BARYCORR=0
INTRODUCED_SHIFT=1
BIN_SIZE=50
CALC_TRANSMISSION=1

usage() {
  cat <<'USAGE'
Usage: scripts/prepare_all.sh [options]

Options:
  --data-dir DIR           Raw data directory (default: input/raw)
  --arms LIST              Comma-separated arms (default: red)
  --barycorr               Enable barycentric correction
  --no-barycorr            Disable barycentric correction (default)
  --no-introduced-shift    Disable epoch-specific Molecfit shift
  --bin-size N             Bin size in pixels (default: 50)
  --no-transmission        Save combined spectrum instead of transmission
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --arms)
      ARMS="$2"
      shift 2
      ;;
    --barycorr)
      BARYCORR=1
      shift
      ;;
    --no-barycorr)
      BARYCORR=0
      shift
      ;;
    --no-introduced-shift)
      INTRODUCED_SHIFT=0
      shift
      ;;
    --bin-size)
      BIN_SIZE="$2"
      shift 2
      ;;
    --no-transmission)
      CALC_TRANSMISSION=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

arms_list=()
IFS=',' read -r -a arms_list <<< "$ARMS"

for dir in "$DATA_DIR"/????????_*; do
  [[ -d "$dir" ]] || continue
  base=$(basename "$dir")
  epoch=${base%%_*}
  planet=${base#*_}

  for arm in "${arms_list[@]}"; do
    cmd=(python -m dataio.make_transmission --epoch "$epoch" --planet "$planet" --arm "$arm" --data-dir "$DATA_DIR" --bin-size "$BIN_SIZE")

    if [[ "$BARYCORR" -eq 1 ]]; then
      cmd+=(--barycorr)
    fi
    if [[ "$INTRODUCED_SHIFT" -eq 0 ]]; then
      cmd+=(--no-introduced-shift)
    fi

    echo "Running: ${cmd[*]}"
    if ! "${cmd[@]}"; then
      echo "FAILED: $epoch $planet $arm (continuing)"
    fi
  done
done
