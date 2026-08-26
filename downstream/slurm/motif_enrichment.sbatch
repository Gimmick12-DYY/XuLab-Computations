#!/bin/bash
#SBATCH --job-name=ds_motif
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#
# Per-TF motif enrichment on the (open-chromatin-masked) imputed calls, run only
# where the TF has a reference motif (downstream/motif_reference_manifest.tsv):
#   HOMER runs when  homer = yes
#   AME   runs when  hocomoco_full is present (HOCOMOCO v11 FULL, mono human)
# TFs with neither are skipped; a TF with only one reference runs only that step.
#
# See eligible TFs and array size, then submit:
#   LIST=1 bash downstream/slurm/motif_enrichment.sbatch
#   N=$(LIST=1 bash downstream/slurm/motif_enrichment.sbatch | grep -vc '^#')
#   sbatch --array=0-$((N-1)) downstream/slurm/motif_enrichment.sbatch
#
# Peaks are called at MACS3 q<0.05 on the imputed pseudobulk (call_imputed_peaks.py,
# reusing peak_coverage.py -- the project peak-calling standard), then HOMER + AME.
#
# Required on the cluster:
#   - conda env with macs3 + numpy/scipy (CONDA_ENV, default 'data_prep')
#   - HOMER (findMotifsGenome.pl) and/or MEME suite (ame, bed2fasta) on PATH
#     (set HOMER_MODULE / MEME_MODULE to `module load` them, or bake into CONDA_ENV)
#   - HOMER genome 'hg38' installed (or HOMER_GENOME=/path/to/hg38.fa)
#   - HOCOMOCO_MEME: HOCOMOCOv11_full_HUMAN_mono_meme_format.meme
#
# Outputs under ${OUT_ROOT}/<tf>/: <tf>.imputed_peaks.bed, <tf>.background.bed,
#   homer/ (if HOMER ran), ame/<tf>/ (if AME ran).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

MANIFEST="${MANIFEST:-${DOWN_DIR}/motif_reference_manifest.tsv}"
[[ -s "${MANIFEST}" ]] || { echo "manifest not found: ${MANIFEST}" >&2; exit 1; }

# TFs with any reference: HOMER (col2=yes) OR HOCOMOCO full (col4 present).
eligible_tfs() {
  awk -F'\t' '
    /^#/ || $1=="tf" || NF<4 { next }
    { h=(tolower($2)=="yes"); a=(toupper($4)!="MISSING" && $4!=""); if(h||a) print $1 }
  ' "${MANIFEST}"
}
# Per-TF steps -> "run_homer run_ame" (1/0); "-1 -1" if TF absent from manifest.
tf_gate() {
  awk -F'\t' -v tf="$1" '
    /^#/ || $1=="tf" { next }
    $1==tf { print (tolower($2)=="yes")?1:0, (toupper($4)!="MISSING" && $4!="")?1:0; f=1; exit }
    END { if(!f) print "-1 -1" }
  ' "${MANIFEST}"
}

if [[ "${LIST:-0}" == "1" ]]; then
  eligible_tfs
  echo "# eligible=$(eligible_tfs | wc -l) (HOMER where homer=yes; AME where hocomoco_full present)" >&2
  exit 0
fi

# ---- TF for this array task ------------------------------------------------
if [[ -n "${TFS:-}" ]]; then
  read -r -a TF_ARR <<< "${TFS}"           # explicit list (still gated by manifest)
else
  mapfile -t TF_ARR < <(eligible_tfs)
fi
(( ${#TF_ARR[@]} > 0 )) || { echo "no eligible TFs in ${MANIFEST}" >&2; exit 1; }
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#TF_ARR[@]} )); then
  echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2
  exit 1
fi
TF="${TF_ARR[$IDX]}"

read -r RUN_HOMER RUN_AME < <(tf_gate "${TF}")
if [[ "${RUN_HOMER}" == "-1" ]]; then
  echo "[skip] ${TF} not in manifest" >&2; exit 0
fi
if (( RUN_HOMER == 0 && RUN_AME == 0 )); then
  echo "[skip] ${TF}: no reference motif" >&2; exit 0
fi

# ---- paths / knobs ---------------------------------------------------------
IMPUTE_DIR="${IMPUTE_DIR:-${ROOT_DIR}/unified/work/${TF}/impute}"

# TF-specific (consensus-subtracted) peaks: call peaks on the residual above the
# shared accessibility consensus, isolating TF-specific signal. Separate output
# tree (motif_tfspec) so it never collides with the plain (accessibility) run.
TF_SPECIFIC="${TF_SPECIFIC:-0}"
CONSENSUS_NPY="${CONSENSUS_NPY:-${DOWN_DIR}/motif/consensus_accessibility.npy}"
CONSENSUS_NORMALIZE="${CONSENSUS_NORMALIZE:-rank}"
if [[ "${TF_SPECIFIC}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT:-${DOWN_DIR}/motif_tfspec}"
else
  OUT_ROOT="${OUT_ROOT:-${DOWN_DIR}/motif}"
fi
OUT="${OUT_ROOT}/${TF}"
HG38_FA="${HG38_FA:-${DOWN_DIR}/cache/hg38.fa}"
HOMER_GENOME="${HOMER_GENOME:-hg38}"
HOCOMOCO_MEME="${HOCOMOCO_MEME:-${DOWN_DIR}/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme}"
OPEN_BED="${OPEN_BED:-${ROOT_DIR}/data/HEK293T_ATAC_union3.bed.gz}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

MACS_Q="${MACS_Q:-0.05}"      # imputed peaks called at MACS3 q<0.05 (project standard)
MIN_PEAKS="${MIN_PEAKS:-200}" # below this, skip motif enrichment (too sparse to validate)
PEAK_WORK="${OUT}/peaks"      # MACS3 synthetic BED + narrowPeak intermediates
# Background: default OFF, matching the original commands exactly --
#   HOMER: findMotifsGenome.pl <peaks> hg38 <out> -useNewBg -p 8 -size given
#   AME:   ame --o <out> <peaks.fa> HOCOMOCO.meme        (shuffled control)
# Opt-in accessibility-matched background (open chromatin minus this TF's peaks,
# <tf>.background.bed) via HOMER_BG=1 / AME_CONTROL=1 -- tried, did not recover
# TF-specific motifs (imputed peaks are ~accessibility, 75-91% shared across TFs).
HOMER_BG="${HOMER_BG:-0}"        # 0 -> HOMER -useNewBg (original); 1 -> -bg <background>
AME_CONTROL="${AME_CONTROL:-0}"  # 0 -> AME shuffled (original); 1 -> --control <background.fa>

BED="${OUT}/${TF}.imputed_peaks.bed"
BG_BED="${OUT}/${TF}.background.bed"
mkdir -p "${OUT}"

# ---- environment -----------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-data_prep}"   # needs macs3 + numpy/scipy (peak calling)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -n "${HOMER_MODULE:-}" ]] && module load ${HOMER_MODULE}
[[ -n "${MEME_MODULE:-}" ]] && module load ${MEME_MODULE}

command -v macs3 >/dev/null 2>&1 || { echo "ERROR: macs3 not on PATH in env ${CONDA_ENV}" >&2; exit 1; }
if (( RUN_HOMER == 1 )); then
  command -v findMotifsGenome.pl >/dev/null 2>&1 || { echo "ERROR: findMotifsGenome.pl not on PATH" >&2; exit 1; }
fi
if (( RUN_AME == 1 )); then
  command -v ame >/dev/null 2>&1 || { echo "ERROR: ame not on PATH" >&2; exit 1; }
  [[ -s "${HOCOMOCO_MEME}" ]] || { echo "ERROR: HOCOMOCO full meme not found: ${HOCOMOCO_MEME}" >&2; exit 1; }
fi

echo "[$(date)] TF=${TF} run_homer=${RUN_HOMER} run_ame=${RUN_AME} tf_specific=${TF_SPECIFIC}"

# Consensus-subtracted (TF-specific) peak calling requires a prebuilt consensus.
CONSENSUS_ARGS=()
if [[ "${TF_SPECIFIC}" == "1" ]]; then
  [[ -s "${CONSENSUS_NPY}" ]] || { echo "ERROR: TF_SPECIFIC=1 but consensus missing: ${CONSENSUS_NPY}
  build once: python ${DOWN_DIR}/build_consensus_accessibility.py --out-npy ${CONSENSUS_NPY}" >&2; exit 1; }
  CONSENSUS_ARGS=(--consensus-npy "${CONSENSUS_NPY}" --consensus-normalize "${CONSENSUS_NORMALIZE}")
fi

# ---- 1. imputed matrix -> q<0.05 peak BED (+ background) --------------------
# Every analyzed TF is peak-called at MACS3 q<MACS_Q first. Reuse is keyed on the
# MACS narrowPeak (produced ONLY by this q<0.05 caller), never on the final BED --
# so a stale BED from any other method is always regenerated, not reused.
NARROW="${PEAK_WORK}/${TF}_qcut_peaks.narrowPeak"
if [[ -s "${NARROW}" && -s "${BED}" ]]; then
  echo "[$(date)] ${TF}: reuse existing q<${MACS_Q} peaks (${NARROW})"
else
  echo "[$(date)] ${TF}: calling peaks at MACS3 q<${MACS_Q}"
  set +e
  python "${DOWN_DIR}/call_imputed_peaks.py" \
    --impute-dir "${IMPUTE_DIR}" --tf "${TF}" \
    --out-bed "${BED}" --out-dir "${PEAK_WORK}" \
    --macs-q "${MACS_Q}" --min-peaks "${MIN_PEAKS}" \
    --open-bed "${OPEN_BED}" --bg-out "${BG_BED}" \
    "${CONSENSUS_ARGS[@]+"${CONSENSUS_ARGS[@]}"}"
  rc=$?
  set -e
  if (( rc == 3 )); then
    echo "[skip] ${TF}: fewer than ${MIN_PEAKS} imputed peaks; not running enrichment" >&2
    echo "${TF}" >> "${OUT_ROOT}/skipped_insufficient_peaks.txt"
    exit 0
  elif (( rc != 0 )); then
    exit "${rc}"
  fi
fi

# Enforce the peak floor on the FINAL BED -- covers the reuse branch too, so a
# stale/small BED (e.g. from the old prevalence method) can never reach HOMER/AME
# and produce a vacuous "Total target sequences = 1" result.
N_PEAKS=$(wc -l < "${BED}" 2>/dev/null | tr -d '[:space:]' || echo 0)
if (( ${N_PEAKS:-0} < MIN_PEAKS )); then
  echo "[skip] ${TF}: ${N_PEAKS} peaks in BED < MIN_PEAKS=${MIN_PEAKS}; not running enrichment" >&2
  echo "${TF}" >> "${OUT_ROOT}/skipped_insufficient_peaks.txt"
  exit 0
fi
echo "[$(date)] ${TF}: ${N_PEAKS} peaks -> enrichment"

# ---- 2. HOMER de novo + known motif enrichment -----------------------------
if (( RUN_HOMER == 1 )); then
  if [[ ! -s "${OUT}/homer/homerResults.html" ]]; then
    echo "[$(date)] ${TF}: HOMER findMotifsGenome.pl"
    HOMER_ARGS=(-p "${THREADS}" -size given)
    if [[ "${HOMER_BG}" == "1" && -s "${BG_BED}" ]]; then
      HOMER_ARGS+=(-bg "${BG_BED}")
    else
      HOMER_ARGS+=(-useNewBg)
    fi
    findMotifsGenome.pl "${BED}" "${HOMER_GENOME}" "${OUT}/homer/" "${HOMER_ARGS[@]}"
  else
    echo "[$(date)] ${TF}: HOMER results exist, skipping"
  fi
else
  echo "[$(date)] ${TF}: no HOMER reference, skipping HOMER"
fi

# ---- 3. AME known-motif enrichment (HOCOMOCO v11 full) ---------------------
if (( RUN_AME == 1 )); then
  mkdir -p "${OUT}/ame/seq"
  PEAK_FA="${OUT}/ame/seq/${TF}.peaks.fa"
  if command -v bed2fasta >/dev/null 2>&1; then
    bed2fasta -o "${PEAK_FA}" "${BED}" "${HG38_FA}"
  else
    echo "[$(date)] bed2fasta not found; using bedtools getfasta"
    bedtools getfasta -fi "${HG38_FA}" -bed "${BED}" -fo "${PEAK_FA}"
  fi

  AME_ARGS=(--o "${OUT}/ame/${TF}")
  if [[ "${AME_CONTROL}" == "1" && -s "${BG_BED}" ]]; then
    BG_FA="${OUT}/ame/seq/${TF}.background.fa"
    if command -v bed2fasta >/dev/null 2>&1; then
      bed2fasta -o "${BG_FA}" "${BG_BED}" "${HG38_FA}"
    else
      bedtools getfasta -fi "${HG38_FA}" -bed "${BG_BED}" -fo "${BG_FA}"
    fi
    AME_ARGS+=(--control "${BG_FA}")
  fi
  echo "[$(date)] ${TF}: AME (HOCOMOCO v11 full)"
  ame "${AME_ARGS[@]}" "${PEAK_FA}" "${HOCOMOCO_MEME}"
else
  echo "[$(date)] ${TF}: not in HOCOMOCO full, skipping AME"
fi

echo "[$(date)] done ${TF} -> ${OUT}"
