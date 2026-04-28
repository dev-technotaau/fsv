#!/usr/bin/env bash
# Run Stage 2 hyperparam search on a specific combo key.
# Usage: ./scripts/ga/run_stage2.sh <combo_key> [extra --set args]
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <combo_key> [extra --set args]"
    echo ""
    echo "Available combo keys:"
    python -m src.ga.cli list-combos
    exit 1
fi

COMBO="$1"
shift
python -m src.ga.cli stage2 "$COMBO" "$@"
