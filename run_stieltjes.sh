#!/usr/bin/env bash
# Run Stieltjes max-retrieval experiments (q=2,4,8,16,32,64) on MPS overnight.
# Usage: bash run_stieltjes.sh
# Results are written to results/ with a timestamp.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$REPO/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG="$RESULTS_DIR/stieltjes_${TIMESTAMP}.log"

echo "Repo:    $REPO"
echo "Log:     $LOG"
echo "Python:  $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "MPS:     $(python3 -c 'import torch; print(torch.backends.mps.is_available())')"
echo ""
echo "Starting experiments at $(date)" | tee "$LOG"
echo "" | tee -a "$LOG"

cd "$REPO"
python3 main.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
echo "Results saved to: $LOG"
