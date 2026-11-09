#!/bin/bash
#SBATCH -J b6_core
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/b6_core/logs/%x_%j.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/regulatome/m4_validation/b6_core; cd $O
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $O/b6_core.py
cat > README.md <<RME
# B6: R5 core lists vs Perturb-seq knockdown (third read)
This is the THIRD read of the X-Atlas/Orion knockdown matrices (after B3/B4 and B5). Job ${SLURM_JOB_ID}, run $(date -u +%FT%TZ).
Spec: SPEC.md (spec + Addendum 1 + Addendum 2, saved from text relays). Final frozen spec sha256 (design thread shared file): 25ac1e76e378700ac7462ecf341dbf71305e3c84af0f6419fc8684490c1e92a2 (supersedes 6fd5766409b0bdb5813fc3326fd1bc08d86d9f9ac7d9c6544d27c6f24113d098; pre-Addendum-2 sha 920789772028a3c1c5de1f76145b94233c173b7919118a542db58d95a297fdc0). Saved copy SPEC.md sha256: $(sha256sum SPEC.md | cut -d' ' -f1); the copy was relayed as text, so its bytes differ from the frozen file.
Addendum 2 written 2026-11-09T19:25:46Z (shared-file timestamp); received on this machine 2026-11-09T19:26:03Z, before any knockdown matrix was read and before any B6 statistic was computed (this job is the first read), so Addendum 2 is co-primary; Addendum 1 effects are primary.
Manifests read: r5_core manifest_v7.sha256 b89b483bba64b4e54cd39ed4f116510528702e3d35ebd9b282b7fe01f45f76ac (verified at start). Knockdown inputs and their sha256: inputs_sha256.tsv.
Lists: lists/<TF>.tsv from b6_lists.py (built before any knockdown read). A TF enters an arm if >= 50 genes of list (a) are in the arm universe; any list with < 10 genes in the universe is not scored for that TF (NA).
Effect (Addendum 1): b3b4.py score() unchanged; signed version = -mean(R over list) / SD(R over the universe), descriptive. Pooled: b3b4.py bstat() (bootstrap over TFs 10,000, two-sided sign test); ceiling weights = b3b4/reliability_ceilings.tsv weight (r <= 0 -> 0), averaged per panel TF. Random-set seed 20261109 (+1 for arm 2), pooled bootstrap seed 20261109.
Addendum 2: responders = top 5% (secondary 10%) of |R| per TF in the arm universe; Fisher exact two-sided OR per list; pooled median OR with bootstrap CI of the median over TFs, fraction OR > 1, and bstat() of log OR(list) - log OR(d) (log OR with +0.5 in each cell so zero cells stay finite; per-TF Fisher OR and p are unmodified).
Diagonal: b3b4.py diagonal test unchanged, on list (a), over KD TFs whose panel TF entered the arm.
Outputs: pooled.tsv, per_tf.tsv, diagonal.tsv, diagonal_arm{1,2}.npz, inputs_sha256.tsv, logs/. manifest_b6.sha256 written last.
RME
find . -type f ! -name manifest_b6.sha256 ! -path './logs/*' | sort | xargs sha256sum > manifest_b6.sha256
sha256sum manifest_b6.sha256
