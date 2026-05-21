# shellcheck shell=bash
# Shared impute paths for downstream/slurm/*.sbatch (source after ROOT_DIR is set).

pipeline_paths_native() {
  MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
  CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
  FITS_DIR="${FITS_DIR:-${ROOT_DIR}/FITS/work/ctcf/impute}"
  SCOPEN_DIR="${SCOPEN_DIR:-${ROOT_DIR}/scOpen/work/ctcf/impute}"
  MAGIC_DIR="${MAGIC_DIR:-${ROOT_DIR}/MAGIC/work/ctcf/impute}"
  PUSCOPEN_DIR="${PUSCOPEN_DIR:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"
  CICERO_DIR="${CICERO_DIR:-${ROOT_DIR}/Cicero/work/ctcf/impute}"
  SCBASSET_DIR="${SCBASSET_DIR:-${ROOT_DIR}/scBasset/work/ctcf/impute}"
  SCBASSET_PUSCOPEN_DIR="${SCBASSET_PUSCOPEN_DIR:-${ROOT_DIR}/scBasset_PUscOpen/work/ctcf/impute}"
  SCBASSET_CISTOPIC_DIR="${SCBASSET_CISTOPIC_DIR:-${ROOT_DIR}/scBasset_cisTopic/work/ctcf/impute}"
}

# Set INPUTS if unset. MODE: all (raw + imputed) | imputed (nine pipelines)
default_inputs() {
  local mode="${1:-imputed}"
  pipeline_paths_native
  local imputed_common=(
    "cisTopic=${CISTOPIC_DIR}"
    "FITS=${FITS_DIR}"
    "scOpen=${SCOPEN_DIR}"
    "MAGIC=${MAGIC_DIR}"
    "PUscOpen=${PUSCOPEN_DIR}"
    "cicero=${CICERO_DIR}"
    "scbasset=${SCBASSET_DIR}"
    "scbasset_puscopen=${SCBASSET_PUSCOPEN_DIR}"
    "scbasset_cistopic=${SCBASSET_CISTOPIC_DIR}"
  )
  case "${mode}" in
    all)
      INPUTS="raw=${MM_DIR} ${imputed_common[*]}"
      ;;
    imputed)
      INPUTS="${imputed_common[*]}"
      ;;
    *)
      echo "default_inputs: unknown mode ${mode} (use: all | imputed)" >&2
      return 1
      ;;
  esac
  export INPUTS
}
