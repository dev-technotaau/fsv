#!/usr/bin/env bash
# Convenience wrapper — loads .env if present, runs scraper.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
    set -a
    . .env
    set +a
fi

python -m data_scraper.cli run --config configs/scraper.yaml "$@"
