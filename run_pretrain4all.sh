#!/bin/bash
# Master launcher to pretrain RegNet on every dataset under data/Benchmark Dataset/

set -e

ROOT="data/Benchmark Dataset"
SCRIPT="scripts/run_pretrain_single.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT not found" >&2; exit 1; fi

# Find leaf directories ending with TFs+*
mapfile -t DATA_DIRS < <(find "$ROOT" -type d -name 'TFs+*' | sort)

echo "Found ${#DATA_DIRS[@]} datasets:"; printf ' - %s\n' "${DATA_DIRS[@]}"

echo "Submitting jobs …"
for dir in "${DATA_DIRS[@]}"; do
  sbatch "$SCRIPT" "$dir"
  sleep 1  # brief delay to avoid scheduler spam
done

echo "All jobs submitted. Use squeue -u $USER to monitor." 