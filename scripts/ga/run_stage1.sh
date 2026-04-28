#!/usr/bin/env bash
# Run Stage 1 model-family search. Usage: ./scripts/ga/run_stage1.sh [--resume]
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="configs/ga_stage1_model_search.yaml"

if [[ "${1:-}" == "--resume" ]]; then
    CKPT="runs/ga/stage1/ga_checkpoint.pkl"
    if [[ ! -f "$CKPT" ]]; then
        echo "No checkpoint at $CKPT"
        exit 1
    fi
    python -m src.ga.cli resume "$CKPT" --config "$CONFIG"
else
    python -m src.ga.cli run --config "$CONFIG" "$@"
fi
