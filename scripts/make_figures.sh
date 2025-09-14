#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_figures.sh
#
# Optional env:
#   DATA_DIR: directory with offline copies of ILPD and German Credit CSVs (if no internet).

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Run the Python script from the correct directory
cd "$SCRIPT_DIR"
python generate_figures_from_xgb.py --outdir ../out ${DATA_DIR:+--data-dir "$DATA_DIR"}

echo "Artifacts written to ../out"