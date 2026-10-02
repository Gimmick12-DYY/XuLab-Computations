#!/bin/bash
#SBATCH --job-name=rank_motifs
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# Novel Motif Finding, Step 3: rank each TF's de-novo motifs by HELD-OUT
# performance (AUROC/AUPRC of pos vs neg bins), NOT by HOMER p-value / IC.
#
# Scoring set = the TF's binding BINS (independent of the pmat PEAKS used for
# de-novo discovery, so it is an out-of-representation held-out set). Candidate
# motifs = HOMER de-novo (-> MEME) + STREME, from motif_denovo_pmat.sbatch.
#
#   TFS="rbbp4 znf507 ..." sbatch --array=0-N downstream/slurm/rank_motifs.sbatch
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

if [[ -n "${TFS:-}" ]]; then read -r -a TF_ARR <<< "${TFS}"; else
  read -r -a TF_ARR <<< "${UNKNOWN_TFS:-rbbp4 znf507 glyr1 mbd3 nme2}"; fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
(( IDX >= 0 && IDX < ${#TF_ARR[@]} )) || { echo "idx ${IDX} OOR (n=${#TF_ARR[@]})" >&2; exit 1; }
TF="${TF_ARR[$IDX]}"; tf="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"

DENOVO_ROOT="${DENOVO_ROOT:-${DOWN_DIR}/motif_pmat}"
DN="${DENOVO_ROOT}/${tf}"
OUT="${OUT_ROOT:-${DOWN_DIR}/motif_ranked}/${tf}"
HG38_FA="${HG38_FA:-${DOWN_DIR}/cache/hg38.fa}"
TOP_N_BINS="${TOP_N_BINS:-10000}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT_DIR}/unified/work/${tf}/mm}"   # TF binding bins (raw)
OPEN_BED="${OPEN_BED:-${ROOT_DIR}/data/HEK293T_ATAC_union3.bed.gz}"  # OCR for matched bg
mkdir -p "${OUT}"

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
MEME_BIN="${MEME_BIN:-/nas/longleaf/rhel9/apps/meme/5.5.7/bin}"
[[ -x "${MEME_BIN}/streme" ]] && export PATH="${MEME_BIN}:${PATH}"

# ---- 1. held-out pos/neg bins (top-N binding bins + OCR-matched background) --
POS_BED="${OUT}/${tf}.pos.bed"; NEG_BED="${OUT}/${tf}.neg.bed"
if [[ ! -s "${POS_BED}" || ! -s "${NEG_BED}" ]]; then
  echo "[$(date)] ${TF}: emit pos/neg bins (top-${TOP_N_BINS})"
  python "${DOWN_DIR}/emit_bins_bed.py" \
    --matrix-dir "${MATRIX_DIR}" --tf "${tf}" --top-n "${TOP_N_BINS}" \
    --out-bed "${POS_BED}" --bg-out "${NEG_BED}" --open-bed "${OPEN_BED}" ${EMIT_EXTRA:-}
fi
[[ -s "${POS_BED}" && -s "${NEG_BED}" ]] || { echo "[skip] ${TF}: no pos/neg bins" >&2; exit 0; }

tofa() { bedtools getfasta -fi "${HG38_FA}" -bed "$1" -fo "$2"; }
tofa "${POS_BED}" "${OUT}/pos.fa"
tofa "${NEG_BED}" "${OUT}/neg.fa"

# ---- 2. assemble candidate motifs: HOMER de-novo (-> MEME) + STREME ----------
CAND="${OUT}/candidates.meme"; : > "${CAND}"
if [[ -s "${DN}/homer/homerMotifs.all.motifs" ]]; then
  python "${DOWN_DIR}/scripts/homer_motifs_to_meme.py" \
    --homer "${DN}/homer/homerMotifs.all.motifs" --out "${OUT}/homer_denovo.meme"
  cat "${OUT}/homer_denovo.meme" >> "${CAND}"
fi
if [[ -s "${DN}/streme/streme.txt" ]]; then
  # streme.txt is MEME format already; append its MOTIF blocks
  awk '/^MEME version/{p=1} p' "${DN}/streme/streme.txt" >> "${CAND}" 2>/dev/null || true
fi
[[ -s "${CAND}" ]] || { echo "[skip] ${TF}: no de-novo motifs in ${DN}" >&2; exit 0; }

# ---- 3. rank by held-out AUROC/AUPRC ----------------------------------------
echo "[$(date)] ${TF}: rank motifs by performance"
python "${DOWN_DIR}/scripts/rank_motifs_by_performance.py" \
  --motifs "${CAND}" --pos-fa "${OUT}/pos.fa" --neg-fa "${OUT}/neg.fa" \
  --out "${OUT}/${tf}.ranked_motifs.tsv"

# ---- 4. Step 2: flag artifacts (TOMTOM vs HOCOMOCO v12 + Codebook) -----------
# shortlist = high AUROC AND pass==True (not a CTCF/NFY/YY1/SP-KLF/ETS contaminant,
# not TE-clustered). TE check auto-enables when the UCSC rmsk BED is present
# (downstream/scripts/fetch_hg38_rmsk_bed.sh) AND INSTANCES_BED (FIMO) is given.
MOTIF_DB="${MOTIF_DB:-${DOWN_DIR}/cache/motifdb/merged_human_motifs.meme}"
REPEAT_BED="${REPEAT_BED:-${DOWN_DIR}/cache/hg38.rmsk.bed.gz}"
[[ -s "${REPEAT_BED}" ]] || REPEAT_BED=""
if [[ -s "${MOTIF_DB}" ]]; then
  tomtom -no-ssc -oc "${OUT}/tomtom" -verbosity 1 -min-overlap 5 -dist pearson -thresh 0.1 \
    "${CAND}" "${MOTIF_DB}" || echo "[warn] tomtom failed for ${TF}" >&2
  if [[ -s "${OUT}/tomtom/tomtom.tsv" ]]; then
    python "${DOWN_DIR}/scripts/flag_motif_artifacts.py" \
      --tomtom "${OUT}/tomtom/tomtom.tsv" --ranked "${OUT}/${tf}.ranked_motifs.tsv" \
      --out "${OUT}/${tf}.ranked_annotated.tsv" \
      ${FAMILIES:+--families "${FAMILIES}"} \
      ${REPEAT_BED:+--repeat-bed "${REPEAT_BED}"} ${INSTANCES_BED:+--instances-bed "${INSTANCES_BED}"}
  fi
else
  echo "[warn] no MOTIF_DB (${MOTIF_DB}); skip Step-2 flagging (build via merge_meme_db.py)" >&2
fi
echo "[$(date)] done ${TF} -> ${OUT}/${tf}.ranked_annotated.tsv"
