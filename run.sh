#!/usr/bin/env bash
set -euo pipefail

MODE="test"

if [[ $# -gt 0 && "$1" == "candidate" ]]; then
    MODE="candidate"
    shift
fi

python scripts/rank.py \
    --mode "$MODE" \
    --data-dir ./data \
    --models-dir ./models \
    --output ./data/final_ranking.csv \
    --top-n 100 \
    "$@"

echo
echo "Done. Output written to ./data/final_ranking.csv"