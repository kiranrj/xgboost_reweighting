#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_figures.sh
#
# Optional env:
#   DATA_DIR: directory with offline copies of ILPD and German Credit CSVs (if no internet).

python generate_figures_from_xgb.py --outdir ../out ${DATA_DIR:+--data-dir "$DATA_DIR"}

echo "Artifacts written to ./out"