#!/bin/bash
#SBATCH --job-name=denovo_pmat
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#
# De novo motif discovery on original pmat peaks (median ~339 bp) for the
# 45 TFs with no Homer/HOCOMOCO motif. HOMER + STREME + TOMTOM vs merged DB.
#
#   TFS="rbbp4 xpa ..." sbatch --array=0-44 downstream/slurm/motif_denovo_pmat.sbatch
#
# Expects <OUT_ROOT>/<tf>/<tf>.pmat_peaks.bed from extract_pmat_peaks.R
# (or extracts on the fly if the BED is missing).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

UNKNOWN_TFS="${UNKNOWN_TFS:-rbbp4 xpa rbbp7 znf507 nme2 mbd3 znf703 glyr1 safb mtf2 hoxb9 flywch1 dr1 pa2g4 znf512b znf644 znf367 ezh2 srcap stag2 wiz znf777 znf746 champ1 pogk thap6 zufsp dnmt1 znf512 zbtb43 adnp trafd1 zbtb40 sox12 akap8l znf581 neurod4 nr1d2 hp1 znf606 znf79 pcgf6 znf503 setdb1 hoxc4}"

if [[ "${LIST:-0}" == "1" ]]; then
  # shellcheck disable=SC2086
  printf '%s\n' ${UNKNOWN_TFS}
  echo "# $(printf '%s\n' ${UNKNOWN_TFS} | wc -l) TFs" >&2
  exit 0
fi

if [[ -n "${TFS:-}" ]]; then
  read -r -a TF_ARR <<< "${TFS}"
else
  read -r -a TF_ARR <<< "${UNKNOWN_TFS}"
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
(( IDX >= 0 && IDX < ${#TF_ARR[@]} )) || { echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2; exit 1; }
TF="${TF_ARR[$IDX]}"

OUT_ROOT="${OUT_ROOT:-${DOWN_DIR}/motif_pmat}"
OUT="${OUT_ROOT}/${TF}"
BED="${OUT}/${TF}.pmat_peaks.bed"
HG38_FA="${HG38_FA:-${DOWN_DIR}/cache/hg38.fa}"
HOMER_GENOME="${HOMER_GENOME:-hg38}"
MOTIF_DB="${MOTIF_DB:-${DOWN_DIR}/cache/motifdb/merged_human_motifs.meme}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
MIN_PEAKS="${MIN_PEAKS:-50}"

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
HOMER_BIN="${HOMER_BIN:-/nas/longleaf/rhel9/apps/homer/5.1/bin}"
[[ -x "${HOMER_BIN}/findMotifsGenome.pl" ]] && export PATH="${HOMER_BIN}:${PATH}"
MEME_BIN="${MEME_BIN:-/nas/longleaf/rhel9/apps/meme/5.5.7/bin}"
[[ -x "${MEME_BIN}/streme" ]] && export PATH="${MEME_BIN}:${PATH}"

mkdir -p "${OUT}"
if [[ ! -s "${BED}" ]]; then
  echo "[$(date)] ${TF}: extracting pmat peaks"
  export PATH="/nas/longleaf/rhel9/apps/r/4.4.0/bin:${PATH}"
  Rscript "${DOWN_DIR}/scripts/extract_pmat_peaks.R" \
    --out-root "${OUT_ROOT}" --tfs "${TF}" --min-cells 2 --n-fg 20000
fi
if [[ ! -s "${BED}" ]]; then
  echo "[skip] ${TF}: no pmat peak BED" >&2
  echo "${TF}" >> "${OUT_ROOT}/skipped_no_peaks.txt"
  exit 0
fi
N_PEAKS=$(wc -l < "${BED}" | tr -d '[:space:]')
if (( N_PEAKS < MIN_PEAKS )); then
  echo "[skip] ${TF}: ${N_PEAKS} peaks < MIN_PEAKS=${MIN_PEAKS}" >&2
  echo "${TF}" >> "${OUT_ROOT}/skipped_insufficient_peaks.txt"
  exit 0
fi
echo "[$(date)] ${TF}: ${N_PEAKS} pmat peaks"

if [[ "${FORCE_HOMER:-0}" == "1" ]]; then rm -rf "${OUT}/homer"; fi
if [[ ! -s "${OUT}/homer/homerResults.html" ]]; then
  echo "[$(date)] ${TF}: HOMER de novo"
  findMotifsGenome.pl "${BED}" "${HOMER_GENOME}" "${OUT}/homer/" \
    -size given -p "${THREADS}" -len "${HOMER_LEN:-8,10,12}" -olen 3 -useNewBg
fi

tofa() { bedtools getfasta -fi "${HG38_FA}" -bed "$1" -fo "$2"; }
FA="${OUT}/${TF}.peaks.fa"
[[ -s "${FA}" ]] || tofa "${BED}" "${FA}"

if [[ "${FORCE_STREME:-0}" == "1" ]]; then rm -rf "${OUT}/streme"; fi
if [[ ! -s "${OUT}/streme/streme.txt" ]]; then
  echo "[$(date)] ${TF}: STREME"
  streme --oc "${OUT}/streme" --dna --p "${FA}" --nmotifs "${STREME_NMOTIFS:-10}" \
    || echo "[warn] STREME failed for ${TF}" >&2
fi

if [[ -s "${MOTIF_DB}" ]]; then
  mkdir -p "${OUT}/tomtom"
  if [[ -s "${OUT}/streme/streme.txt" && ! -s "${OUT}/tomtom/streme/tomtom.tsv" ]]; then
    tomtom -no-ssc -oc "${OUT}/tomtom/streme" -verbosity 1 -min-overlap 5 \
      -dist pearson -thresh 0.1 "${OUT}/streme/streme.txt" "${MOTIF_DB}" \
      || echo "[warn] tomtom(streme) failed" >&2
  fi
  if [[ -s "${OUT}/homer/homerMotifs.all.motifs" && ! -s "${OUT}/tomtom/homer/tomtom.tsv" ]]; then
    python "${DOWN_DIR}/scripts/homer_motifs_to_meme.py" \
      --homer "${OUT}/homer/homerMotifs.all.motifs" \
      --out "${OUT}/tomtom/homer_denovo.meme"
    tomtom -no-ssc -oc "${OUT}/tomtom/homer" -verbosity 1 -min-overlap 5 \
      -dist pearson -thresh 0.1 "${OUT}/tomtom/homer_denovo.meme" "${MOTIF_DB}" \
      || echo "[warn] tomtom(homer) failed" >&2
  fi
fi

echo "[$(date)] done ${TF} -> ${OUT}"
