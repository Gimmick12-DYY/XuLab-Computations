#!/bin/bash
#SBATCH --job-name=cx_motif
#SBATCH --output=cx_motif-%A_%a.out
#SBATCH --error=cx_motif-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Step 3 (checklist step 4): shared-motif test per candidate complex. Build each
# complex's co-bound BED + union-peak background (build_complex_beds.py), then
# HOMER de novo + AME (HOCOMOCO v11 full) -- same commands/caches as
# downstream/slurm/motif_enrichment.sbatch. Array = one complex per task.
#
#   # after tf_complexes.py wrote candidate_complexes.tsv:
#   sbatch tf_complex/slurm/03_complex_motifs.sbatch                 # builds BEDs on task 0 too
#   N=$(tail -n +2 tf_complex/results/complex_beds/complex_manifest.tsv | wc -l)
#   sbatch --array=0-$((N-1)) tf_complex/slurm/03_complex_motifs.sbatch
#
# Question: do a complex's co-bound regions share ONE motif (a member's, or a common
# one)? Background = other bound loci, so hits are complex-specific, not generic GC.
#
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
TFC="${ROOT_DIR}/tf_complex"
DOWN_DIR="${ROOT_DIR}/downstream"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work}"
COMPLEXES="${COMPLEXES:-${TFC}/results/candidate_complexes.tsv}"
BED_DIR="${BED_DIR:-${TFC}/results/complex_beds}"
OUT_ROOT="${OUT_ROOT:-${TFC}/results/complex_motifs}"
MIN_FRAC="${MIN_FRAC:-0.5}"

# reuse downstream motif caches
HG38_FA="${HG38_FA:-${DOWN_DIR}/cache/hg38.fa}"
HOMER_GENOME="${HOMER_GENOME:-hg38}"
HOCOMOCO_MEME="${HOCOMOCO_MEME:-${DOWN_DIR}/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
USE_GENOME_BG="${USE_GENOME_BG:-0}"   # 1 -> HOMER -useNewBg / AME shuffled (generic); 0 -> union-peak bg

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -n "${HOMER_MODULE:-}" ]] && module load ${HOMER_MODULE}
[[ -n "${MEME_MODULE:-}" ]] && module load ${MEME_MODULE}

# ---- build the per-complex BEDs once (idempotent) --------------------------
mkdir -p "${BED_DIR}" "${OUT_ROOT}"
MANIFEST="${BED_DIR}/complex_manifest.tsv"
if [[ ! -s "${MANIFEST}" ]]; then
  [[ -s "${COMPLEXES}" ]] || { echo "no candidate_complexes.tsv at ${COMPLEXES}" >&2; exit 1; }
  python "${TFC}/scripts/build_complex_beds.py" \
    --matrix-dir "${MATRIX_DIR}" --complexes "${COMPLEXES}" \
    --out-dir "${BED_DIR}" --min-frac "${MIN_FRAC}"
fi

mapfile -t CIDS < <(tail -n +2 "${MANIFEST}" | cut -f1)
(( ${#CIDS[@]} > 0 )) || { echo "no complexes in ${MANIFEST}" >&2; exit 1; }
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#CIDS[@]} )); then echo "[skip] task ${IDX} >= ${#CIDS[@]} complexes"; exit 0; fi
CID="${CIDS[$IDX]}"
FG="${BED_DIR}/${CID}.cobound.bed"
BG="${BED_DIR}/${CID}.bg.bed"
OUT="${OUT_ROOT}/${CID}"
mkdir -p "${OUT}"
echo "[$(date)] ${CID}: $(wc -l < "${FG}") co-bound loci"

# ---- HOMER de novo + known ------------------------------------------------
if command -v findMotifsGenome.pl >/dev/null 2>&1; then
  if [[ ! -s "${OUT}/homer/homerResults.html" ]]; then
    HOMER_ARGS=(-p "${THREADS}" -size given)
    if [[ "${USE_GENOME_BG}" == "1" ]]; then HOMER_ARGS+=(-useNewBg); else HOMER_ARGS+=(-bg "${BG}"); fi
    findMotifsGenome.pl "${FG}" "${HOMER_GENOME}" "${OUT}/homer/" "${HOMER_ARGS[@]}"
  else echo "[$(date)] ${CID}: HOMER exists, skip"; fi
else echo "[warn] findMotifsGenome.pl not on PATH; skipping HOMER"; fi

# ---- AME (HOCOMOCO v11 full) ----------------------------------------------
if command -v ame >/dev/null 2>&1 && [[ -s "${HOCOMOCO_MEME}" ]]; then
  mkdir -p "${OUT}/ame/seq"
  FG_FA="${OUT}/ame/seq/${CID}.fa"
  tofasta() { if command -v bed2fasta >/dev/null 2>&1; then bed2fasta -o "$2" "$1" "${HG38_FA}"; else bedtools getfasta -fi "${HG38_FA}" -bed "$1" -fo "$2"; fi; }
  tofasta "${FG}" "${FG_FA}"
  AME_ARGS=(--o "${OUT}/ame")
  if [[ "${USE_GENOME_BG}" != "1" ]]; then
    BG_FA="${OUT}/ame/seq/${CID}.bg.fa"; tofasta "${BG}" "${BG_FA}"; AME_ARGS+=(--control "${BG_FA}")
  fi
  ame "${AME_ARGS[@]}" "${FG_FA}" "${HOCOMOCO_MEME}"
else echo "[warn] ame/HOCOMOCO not available; skipping AME"; fi

echo "[$(date)] done ${CID} -> ${OUT}"
