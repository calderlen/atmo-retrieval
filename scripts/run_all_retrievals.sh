#!/usr/bin/env bash
set -euo pipefail

SPECTRA_DIR="input/spectra"
MODE="transmission"
WAVELENGTH_RANGE=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_all_retrievals.sh [options]

Options:
  --spectra-dir DIR     Base spectra directory (default: input/spectra)
  --mode MODE           Retrieval mode (default: transmission)
  --extra-args "..."     Extra args passed to __main__.py
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spectra-dir)
      SPECTRA_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --extra-args)
      IFS=' ' read -r -a EXTRA_ARGS <<< "$2"
      shift 2
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

find "$SPECTRA_DIR" -type f -name "wavelength_${MODE}.npy" | while read -r wav_path; do
  data_dir=$(dirname "$wav_path")
  arm=$(basename "$data_dir")
  epoch=$(basename "$(dirname "$data_dir")")
  planet_dir=$(basename "$(dirname "$(dirname "$data_dir")")")

  map_out=$(python - <<'PY' "$planet_dir"
import sys
from config import PLANETS, EPHEMERIS

planet_dir = sys.argv[1]
planet = None
for name in PLANETS:
    if name.lower().replace('-', '') == planet_dir:
        planet = name
        break
if planet is None:
    sys.exit(2)

if EPHEMERIS in PLANETS[planet]:
    ephem = EPHEMERIS
else:
    ephem = list(PLANETS[planet].keys())[0]
print(f"{planet}|{ephem}")
PY
  ) || {
    echo "Skipping $planet_dir/$epoch/$arm: no matching planet in config"
    continue
  }

  IFS='|' read -r planet ephem <<< "$map_out"

  out_dir="output/${planet_dir}/${ephem}/${MODE}/${epoch}/${arm}"

  cmd=(python __main__.py --mode "$MODE" --planet "$planet" --ephemeris "$ephem" --data-dir "$data_dir" --wavelength-range "$arm" --output "$out_dir")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "Running: ${cmd[*]}"
  if ! "${cmd[@]}"; then
    echo "FAILED: $planet/$epoch/$arm"
  fi

done
