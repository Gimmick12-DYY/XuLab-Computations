#!/bin/bash
#SBATCH --job-name=hic_compartments
#SBATCH --output=hic_compartments-%j.out
#SBATCH --error=hic_compartments-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# A/B compartments from the balanced Hi-C matrix (cooltools eigs-cis), GC-phased
# so E1 > 0 = A (GC-rich) and E1 < 0 = B. This is the bridge from the Hi-C
# pipeline to the co-binding analysis.
#
#   sbatch hic/slurm/04_compartments.sbatch                    # 25 kb (default)
#   COMPARTMENT_RES=100000 sbatch hic/slurm/04_compartments.sbatch
#
# Prereq: 02_pileup produced a balanced .mcool. If it lacks the ${COMPARTMENT_RES}
# level, this coarsens+balances it from a divisor resolution automatically.

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

command -v cooltools >/dev/null 2>&1 || { echo "cooltools not on PATH in env ${CONDA_ENV}" >&2; exit 1; }

MCOOL="${MCOOL:-$(ls "${WORK}"/coolers-"${GENOME}"/*.mcool 2>/dev/null | head -1)}"
[[ -s "${MCOOL:-}" ]] || { echo "no .mcool under ${WORK}/coolers-${GENOME}/ (run 02)" >&2; exit 1; }
RES="${COMPARTMENT_RES}"

OUT="${WORK}/compartments"
mkdir -p "${OUT}"
cd "${OUT}"

COOL="${MCOOL}::/resolutions/${RES}"
if ! cooler info "${COOL}" >/dev/null 2>&1; then
  echo "[$(date)] ${RES} bp not in mcool; available: $(cooler ls "${MCOOL}" | sed 's#.*/##' | tr '\n' ' ')"
  # Largest available resolution that evenly divides RES -> coarsen from it.
  BASE=""
  for r in $(cooler ls "${MCOOL}" | sed 's#.*/##' | sort -nr); do
    [[ "${r}" =~ ^[0-9]+$ ]] || continue
    if (( r <= RES )) && (( RES % r == 0 )); then BASE="${r}"; break; fi
  done
  [[ -n "${BASE}" ]] || { echo "no resolution divides ${RES}; re-bin at ${RES} in step 02." >&2; exit 1; }
  FACTOR=$(( RES / BASE ))
  COARSE="${OUT}/${GENOME}_${RES}.cool"
  echo "[$(date)] coarsen ${BASE} -> ${RES} (x${FACTOR}) + balance"
  cooler coarsen -k "${FACTOR}" -o "${COARSE}" "${MCOOL}::/resolutions/${BASE}"
  cooler balance -p "${THREADS}" "${COARSE}"
  COOL="${COARSE}"
fi

echo "[$(date)] binnify + GC phasing track @ ${RES} bp"
cooltools genome binnify "${CHROMSIZES}" "${RES}" > "bins_${RES}.bed"
cooltools genome gc "bins_${RES}.bed" "${GENOME_FA}" > "gc_${RES}.tsv"

echo "[$(date)] eigs-cis (GC-phased) on ${COOL}"
cooltools eigs-cis \
  --phasing-track "gc_${RES}.tsv" \
  --n-eigs 3 \
  -o "compartments_${RES}" \
  "${COOL}"

# E1 -> A/B bedGraph (col: chrom start end E1). E1>0 = A, E1<0 = B (GC-phased).
VECS="compartments_${RES}.cis.vecs.tsv"
if [[ -s "${VECS}" ]]; then
  python - "$VECS" "compartments_${RES}.E1.bedGraph" <<'PY'
import sys, csv
inp, out = sys.argv[1], sys.argv[2]
with open(inp) as f, open(out, "w") as o:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        e1 = row.get("E1", "")
        if e1 in ("", "nan", "NaN"):
            continue
        o.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{e1}\n")
print("wrote", out)
PY
fi
echo "[$(date)] done -> ${OUT} (compartments_${RES}.cis.vecs.tsv, *.E1.bedGraph)"
