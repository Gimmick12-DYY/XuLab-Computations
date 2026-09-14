#!/bin/bash
#SBATCH --job-name=motif_spec
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# TF-specific de novo motif discovery: panel-differential regions instead of
# "peaks vs genome/OCR".
#
# Motivation (see tf_specific_regions.py for the full argument): the imputed
# calls are 1 kb bins masked to the ATAC union, so peaks are open chromatin and
# summits are meaningless. Every previous background (genome, GC-matched OCR)
# therefore rediscovered promoter grammar. Here the foreground is bins high for
# THIS TF and low across the other 77, and the background is bins equally bound
# by the panel but not by this TF -- so accessibility cancels.
#
# Steps per TF:
#   1. tf_specific_regions.py -> <tf>.specific_fg.bed / <tf>.specific_bg.bed
#   2. HOMER de novo + known, -bg = the matched background
#   3. STREME (discriminative, hold-out significance) fg vs bg
#   4. TOMTOM: de novo motifs vs HOCOMOCO v11 + v12 + JASPAR 2024
#   5. SEA: known-motif enrichment vs the same matched background
#
# Submit:
#   N=$(LIST=1 bash downstream/slurm/motif_denovo_specific.sbatch | grep -vc '^#')
#   sbatch --array=0-$((N-1)) downstream/slurm/motif_denovo_specific.sbatch
# One TF:
#   TFS="rbbp4" sbatch --array=0-0 downstream/slurm/motif_denovo_specific.sbatch
#
# Outputs under downstream/motif_specific/<tf>/: regions/, homer/, streme/,
#   tomtom/, sea/.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

# Prefer the original peak-level panel (median ~339 bp MACS peaks from
# TF1000cells.pmat.mtx.rds). Fall back to the 1 kb imputed panel if the
# peak panel has not been built yet.
if [[ -z "${PANEL_NPZ:-}" ]]; then
  if [[ -s "${DOWN_DIR}/motif/pmat_panel.npz" ]]; then
    PANEL_NPZ="${DOWN_DIR}/motif/pmat_panel.npz"
  else
    PANEL_NPZ="${DOWN_DIR}/motif/panel_prevalence.npz"
  fi
fi
OUT_ROOT="${OUT_ROOT:-${DOWN_DIR}/motif_specific}"
HG38_FA="${HG38_FA:-${DOWN_DIR}/cache/hg38.fa}"
HOMER_GENOME="${HOMER_GENOME:-hg38}"
MOTIF_DB="${MOTIF_DB:-${DOWN_DIR}/cache/motifdb/merged_human_motifs.meme}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

N_FG="${N_FG:-20000}"
BG_RATIO="${BG_RATIO:-2.0}"
HOMER_LEN="${HOMER_LEN:-8,10,12}"
STREME_NMOTIFS="${STREME_NMOTIFS:-10}"

panel_tfs() {
  python - "$PANEL_NPZ" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    for t in z["tfs"]:
        print(str(t))
PY
}

if [[ "${LIST:-0}" == "1" ]]; then
  panel_tfs
  echo "# $(panel_tfs | wc -l) TFs in ${PANEL_NPZ}" >&2
  exit 0
fi

# ---- TF for this array task ------------------------------------------------
if [[ -n "${TFS:-}" ]]; then
  read -r -a TF_ARR <<< "${TFS}"
else
  mapfile -t TF_ARR < <(panel_tfs)
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
(( IDX >= 0 && IDX < ${#TF_ARR[@]} )) || { echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2; exit 1; }
TF="${TF_ARR[$IDX]}"

OUT="${OUT_ROOT}/${TF}"
REG="${OUT}/regions"
mkdir -p "${REG}"

# ---- environment -----------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
HOMER_BIN="${HOMER_BIN:-/nas/longleaf/rhel9/apps/homer/5.1/bin}"
[[ -x "${HOMER_BIN}/findMotifsGenome.pl" ]] && export PATH="${HOMER_BIN}:${PATH}"
MEME_BIN="${MEME_BIN:-/nas/longleaf/rhel9/apps/meme/5.5.7/bin}"
[[ -x "${MEME_BIN}/streme" ]] && export PATH="${MEME_BIN}:${PATH}"

[[ -s "${PANEL_NPZ}" ]] || { echo "ERROR: panel matrix missing: ${PANEL_NPZ}
  build it first: sbatch downstream/slurm/build_panel_matrix.sbatch" >&2; exit 1; }

echo "[$(date)] TF=${TF}"

# Complex partners / paralogs must not sit in the contrast: they share targets,
# so leaving them in cancels exactly the signal we are looking for.
declare -A PARTNERS=(
  [rbbp4]="rbbp7 mbd3 mtf2 ezh2 pcgf6"
  [rbbp7]="rbbp4 mbd3 mtf2 ezh2 pcgf6"
  [mbd3]="rbbp4 rbbp7"
  [ezh2]="mtf2 pcgf6 rbbp4 rbbp7"
  [mtf2]="ezh2 pcgf6 rbbp4 rbbp7"
  [pcgf6]="ezh2 mtf2"
  [setdb1]="hp1 wiz trim28"
  [hp1]="setdb1 wiz"
  [wiz]="setdb1 hp1"
  [ctcf]="stag2"
  [stag2]="ctcf"
  [mef2a]="mef2d"
  [mef2d]="mef2a"
  [znf503]="znf703"
  [znf703]="znf503"
)
EXCLUDE="${EXCLUDE:-${PARTNERS[$TF]:-}}"
EXCL_ARGS=()
[[ -n "${EXCLUDE}" ]] && EXCL_ARGS=(--exclude-others ${EXCLUDE})

# ---- 1. panel-differential regions -----------------------------------------
FG="${REG}/${TF}.specific_fg.bed"
BG="${REG}/${TF}.specific_bg.bed"
if [[ "${FORCE_REGIONS:-0}" == "1" || ! -s "${FG}" ]]; then
  set +e
  python "${DOWN_DIR}/scripts/tf_specific_regions.py" \
    --panel-npz "${PANEL_NPZ}" --tf "${TF}" --out-dir "${REG}" \
    --n-fg "${N_FG}" --bg-ratio "${BG_RATIO}" "${EXCL_ARGS[@]+"${EXCL_ARGS[@]}"}"
  rc=$?; set -e
  if (( rc == 3 )); then
    echo "[skip] ${TF}: no TF-specific bins" >&2
    echo "${TF}" >> "${OUT_ROOT}/skipped_no_specific_bins.txt"; exit 0
  elif (( rc != 0 )); then exit "${rc}"; fi
fi
echo "[$(date)] ${TF}: fg=$(wc -l < "${FG}") bg=$(wc -l < "${BG}")"

# ---- 2. HOMER de novo + known against the matched background ---------------
# -size given keeps the whole 1 kb bin: with no fragment-level data the site can
# be anywhere inside it, so recentering would just throw sequence away.
# -olen 3 suppresses the low-complexity poly-G/CpG motifs that dominated before.
if [[ "${FORCE_HOMER:-0}" == "1" ]]; then rm -rf "${OUT}/homer"; fi
if [[ ! -s "${OUT}/homer/homerResults.html" ]]; then
  echo "[$(date)] ${TF}: HOMER de novo (bg = panel-matched)"
  findMotifsGenome.pl "${FG}" "${HOMER_GENOME}" "${OUT}/homer/" \
    -bg "${BG}" -size given -p "${THREADS}" -len "${HOMER_LEN}" -olen 3
else
  echo "[$(date)] ${TF}: HOMER results exist, skipping"
fi

# ---- 3. STREME discriminative discovery ------------------------------------
tofa() { bedtools getfasta -fi "${HG38_FA}" -bed "$1" -fo "$2"; }
FG_FA="${REG}/${TF}.fg.fa"; BG_FA="${REG}/${TF}.bg.fa"
[[ -s "${FG_FA}" ]] || tofa "${FG}" "${FG_FA}"
[[ -s "${BG_FA}" ]] || tofa "${BG}" "${BG_FA}"

if [[ "${FORCE_STREME:-0}" == "1" ]]; then rm -rf "${OUT}/streme"; fi
if [[ ! -s "${OUT}/streme/streme.txt" ]]; then
  echo "[$(date)] ${TF}: STREME"
  streme --oc "${OUT}/streme" --dna --p "${FG_FA}" --n "${BG_FA}" \
    --nmotifs "${STREME_NMOTIFS}" || echo "[warn] STREME failed for ${TF}" >&2
fi

# ---- 4. annotate de novo motifs (TOMTOM vs v11 + v12 + JASPAR) -------------
if [[ -s "${MOTIF_DB}" ]]; then
  mkdir -p "${OUT}/tomtom"
  if [[ -s "${OUT}/streme/streme.txt" ]]; then
    tomtom -no-ssc -oc "${OUT}/tomtom/streme" -verbosity 1 -min-overlap 5 \
      -dist pearson -thresh 0.1 "${OUT}/streme/streme.txt" "${MOTIF_DB}" \
      || echo "[warn] tomtom(streme) failed for ${TF}" >&2
  fi
  if [[ -s "${OUT}/homer/homerMotifs.all.motifs" ]]; then
    python "${DOWN_DIR}/scripts/homer_motifs_to_meme.py" \
      --homer "${OUT}/homer/homerMotifs.all.motifs" \
      --out "${OUT}/tomtom/homer_denovo.meme"
    tomtom -no-ssc -oc "${OUT}/tomtom/homer" -verbosity 1 -min-overlap 5 \
      -dist pearson -thresh 0.1 "${OUT}/tomtom/homer_denovo.meme" "${MOTIF_DB}" \
      || echo "[warn] tomtom(homer) failed for ${TF}" >&2
  fi

  # ---- 5. known-motif enrichment vs the SAME matched background ------------
  if [[ "${FORCE_SEA:-0}" == "1" ]]; then rm -rf "${OUT}/sea"; fi
  if [[ ! -s "${OUT}/sea/sea.tsv" ]]; then
    echo "[$(date)] ${TF}: SEA (merged DB, panel-matched control)"
    sea --p "${FG_FA}" --n "${BG_FA}" --m "${MOTIF_DB}" --oc "${OUT}/sea" \
      || echo "[warn] SEA failed for ${TF}" >&2
  fi
else
  echo "[warn] motif DB missing (${MOTIF_DB}); skipping TOMTOM/SEA" >&2
fi

echo "[$(date)] done ${TF} -> ${OUT}"
