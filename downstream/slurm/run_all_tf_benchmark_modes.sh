#!/bin/bash
# Submit all three benchmark modes for every TF in the panel.
#
# Usage:
#   bash downstream/slurm/run_all_tf_benchmark_modes.sh
#   DRY_RUN=1 bash downstream/slurm/run_all_tf_benchmark_modes.sh   # print only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_tf_benchmark_modes.sh"

TFS=(CTCF ZBTB7A ZIC2 MAZ ZNF777 NFYA ZNF282)

for tf in "${TFS[@]}"; do
  echo "========== ${tf} =========="
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] bash ${RUNNER} ${tf}"
  else
    bash "${RUNNER}" "${tf}"
  fi
done

echo "[submit] Queued ${#TFS[@]} TFs x 3 modes = $(( ${#TFS[@]} * 3 )) jobs."
