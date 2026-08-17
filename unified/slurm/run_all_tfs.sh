#!/bin/bash
# Extract + unified ingest/train/impute for every TF in TF1000cells.meta.csv.
#
# Skips TFs that already have unified/work/<tf>/impute/matrix_csr.npz
# (override with FORCE=1). Optionally skip named TFs via SKIP_TFS.
#
# Usage:
#   bash unified/slurm/run_all_tfs.sh
#   DRY_RUN=1 bash unified/slurm/run_all_tfs.sh
#   FORCE=1 bash unified/slurm/run_all_tfs.sh MAZ ZIC2
#   bash unified/slurm/run_all_tfs.sh RBBP4 SOX4   # only these
#   SKIP_TFS="ZNF768 SETDB1" bash unified/slurm/run_all_tfs.sh
#
# Env:
#   RUN_EVAL=0|1          (default 0 — impute only)
#   FORCE=0|1             re-run even if impute exists
#   DRY_RUN=0|1
#   SKIP_TFS="A B C"
#   MIN_CELLS=0           skip TFs with fewer cells than this
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${ROOT}/data"
META="${META:-${DATA}/TF1000cells.meta.csv}"
EXTRACT_SBATCH="${ROOT}/data_prep/slurm/extract_tf_matrix.sbatch"
PIPE_SBATCH="${ROOT}/unified/slurm/99_full_pipeline.sbatch"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
MIN_CELLS="${MIN_CELLS:-0}"
SKIP_TFS="${SKIP_TFS:-}"

sbatch_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] sbatch $*" >&2
    # fake parsable id on stdout only (for capture)
    echo "dry$$"
  else
    sbatch --parsable "$@"
  fi
}

should_skip() {
  local tf="$1"
  for s in ${SKIP_TFS}; do
    [[ "${s}" == "${tf}" ]] && return 0
  done
  return 1
}

if [[ ! -f "${META}" ]]; then
  echo "ERROR: missing ${META}" >&2
  exit 1
fi

# Build TF list: either CLI args or all TFs in meta (sorted by cell count asc).
declare -a TFS=()
declare -A N_CELLS=()
while IFS= read -r line; do
  n="${line%% *}"
  tf="${line#* }"
  N_CELLS["${tf}"]="${n}"
done < <(awk -F, '
  NR==1 {
    for (i=1;i<=NF;i++) if (tolower($i)=="tf") c=i
    next
  }
  { a[$c]++ }
  END { for (k in a) print a[k], k }
' "${META}" | sort -n)

if [[ "$#" -gt 0 ]]; then
  TFS=("$@")
else
  # stable ascending by cell count
  while IFS= read -r line; do
    TFS+=("${line#* }")
  done < <(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if (tolower($i)=="tf") c=i; next }
    { a[$c]++ }
    END { for (k in a) print a[k], k }
  ' "${META}" | sort -n)
fi

echo "[plan] ROOT=${ROOT}"
echo "[plan] FORCE=${FORCE} RUN_EVAL=${RUN_EVAL} DRY_RUN=${DRY_RUN} MIN_CELLS=${MIN_CELLS}"
echo "[plan] SKIP_TFS=${SKIP_TFS:-"(none)"}"
echo "[plan] candidates=${#TFS[@]}"

n_submit=0
n_skip=0
declare -a SUBMITTED=()

for TF in "${TFS[@]}"; do
  tf_lc="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
  n="${N_CELLS[${TF}]:-0}"
  imp="${ROOT}/unified/work/${tf_lc}/impute/matrix_csr.npz"
  rds="${DATA}/${TF}_bin1000_mtx.rds"
  cfg="${ROOT}/unified/configs/${tf_lc}.yaml"
  # CTCF historically uses default.yaml
  if [[ "${TF}" == "CTCF" && ! -f "${cfg}" ]]; then
    cfg="${ROOT}/unified/configs/default.yaml"
  fi

  if should_skip "${TF}"; then
    echo "[skip] ${TF}: in SKIP_TFS"
    n_skip=$((n_skip + 1))
    continue
  fi
  if [[ "${n}" -lt "${MIN_CELLS}" ]]; then
    echo "[skip] ${TF}: n_cells=${n} < MIN_CELLS=${MIN_CELLS}"
    n_skip=$((n_skip + 1))
    continue
  fi
  if [[ "${FORCE}" != "1" && -f "${imp}" ]]; then
    echo "[skip] ${TF}: impute exists (${imp})"
    n_skip=$((n_skip + 1))
    continue
  fi

  # Resource scaling by cell count (cell bank + denser batches).
  mem="16G"
  time_lim="08:00:00"
  if [[ "${n}" -ge 20000 ]]; then
    mem="32G"
    time_lim="12:00:00"
  fi
  if [[ "${n}" -ge 80000 ]]; then
    mem="64G"
    time_lim="24:00:00"
  fi

  dep=()
  if [[ ! -f "${rds}" ]]; then
    echo "[submit] extract TF=${TF} (n_cells=${n})"
    extract_job="$(sbatch_or_echo --job-name="extract_${tf_lc}" \
      --export=ALL,TF="${TF}",MAKE_UNIFIED_CONFIG=1 \
      "${EXTRACT_SBATCH}")"
    dep=(--dependency="afterok:${extract_job}")
    cfg="${ROOT}/unified/configs/${tf_lc}.yaml"
  elif [[ ! -f "${cfg}" && "${TF}" != "CTCF" ]]; then
    echo "[submit] extract TF=${TF} (rds exists but config missing)"
    extract_job="$(sbatch_or_echo --job-name="extract_${tf_lc}" \
      --export=ALL,TF="${TF}",MAKE_UNIFIED_CONFIG=1 \
      "${EXTRACT_SBATCH}")"
    dep=(--dependency="afterok:${extract_job}")
    cfg="${ROOT}/unified/configs/${tf_lc}.yaml"
  fi

  skip_export=0
  if [[ -f "${ROOT}/unified/work/${tf_lc}/mm/matrix.mtx.gz" ]]; then
    skip_export=1
  fi

  echo "[submit] unified TF=${TF} n_cells=${n} mem=${mem} time=${time_lim} cfg=${cfg} SKIP_EXPORT=${skip_export}"
  # Always submit from ROOT so SLURM_SUBMIT_DIR resolves paths correctly
  # (sbatch copies the script to /var/spool; BASH_SOURCE fallback is unsafe).
  pipe_job="$(sbatch_or_echo "${dep[@]}" \
    --chdir="${ROOT}/unified" \
    --job-name="unified_${tf_lc}" \
    --mem="${mem}" \
    --time="${time_lim}" \
    --export=ALL,CFG="${cfg}",RUN_EVAL="${RUN_EVAL}",SKIP_EXPORT="${skip_export}" \
    "${PIPE_SBATCH}")"
  SUBMITTED+=("${TF}:${pipe_job}:n=${n}")
  n_submit=$((n_submit + 1))
done

echo
echo "[done] submitted=${n_submit} skipped=${n_skip}"
if [[ "${#SUBMITTED[@]}" -gt 0 ]]; then
  printf '  %s\n' "${SUBMITTED[@]}"
fi
if [[ "${DRY_RUN}" != "1" && "${n_submit}" -gt 0 ]]; then
  squeue -u "${USER}" -o '%.18i %.12P %.20j %.2t %.10M %.6D %R' | head -60
fi
