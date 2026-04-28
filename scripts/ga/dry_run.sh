#!/usr/bin/env bash
# Fast sanity check — validates config + adapter imports + loops GA with random fitness.
# Usage: ./scripts/ga/dry_run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m src.ga.cli dry-run --config configs/ga_stage1_model_search.yaml \
    --set ga.population_size=6 \
    --set ga.generations=2 \
    --set runtime.output_dir=runs/ga/dry_run
