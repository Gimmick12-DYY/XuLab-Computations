WHAT IS READ: the existing m4_validation/kd_response/out_v2/ matrices only.
- Arm 1: high-KD, 30 TFs.
- Arm 2: all-cell, 77 TFs / 84 constructs, averaged per TF.
- Double-centred within the ≥10% NTC-detection universe. No new cell selection or KD processing.
- This is the THIRD read (after B3/B4 and B5).

LISTS: from r5_core. Verify manifest_v7 (b89b483b…) FIRST. Per TF:
- (a) core top_1000 flag
- (b) promoter-only top 1000, ranked by promoter-element contributions alone
- (c) distal-only top 1000, ranked by distal contributions alone
- (d) S_atac top 1000 (accessibility-only null)
- (e) binding-only null: top 1000 by summed raw CPM at the gene's linked elements, no ABC weighting

Intersect each list with the arm's gene universe. A TF enters an arm only if ≥50 genes of (a) remain.

EFFECT: mean double-centred KD response over the list genes, signed as in B3/B4 (repression of targets by the TF = positive), in SD units of the TF's response vector.

PRIMARY, per arm:
- Pooled mean of effect(a) − effect(d).
- Bootstrap over TFs: 10,000 draws, seed 20261109, 95% CI.
- Two-sided sign test on the per-TF differences.
- Also report (a) vs 0 and (a) − (e).

SECONDARY:
- The same statistics for (b) and for (c).
- Paired (b) − (c).
- Ceiling-weighted versions using the b3b4 reliability_ceilings.tsv weights (replicate r ≤ 0 → weight 0).

DESCRIPTIVE:
- Per-TF effects for (a)–(e), with no per-TF p-values.
- Diagonal test on the core S matrix exactly as in B3: own-KD vs off-KD, permutation p.

No other statistic. No changes after freezing. No multiplicity correction beyond reporting both arms.

OUTPUTS: m4_validation/b6_core/
- pooled.tsv, per_tf.tsv, diagonal.tsv
- README.md stating "third read", the manifest hashes read and the spec sha
- manifest_b6 written LAST

Scripts go under regulatome/m4_validation/b6_core/ (reuse b3b4 code wherever possible), committed by name under the user identity, no attribution. Resources: 1 CPU, 16 GB, <15 min.

Addendum 1: The Per-TF effect line above is replaced by the B3/B4 definition as coded in b3b4.py, so that B6 is on the same scale as B3/B4/B5. Effect = mean absolute double-centred response |R| over list genes, minus the mean of the same quantity over 1,000 expression-matched random gene sets (20 bins of NTC mean), divided by the SD of those random sets. Unsigned. The accessibility-only and binding-only nulls are scored identically. SECONDARY, descriptive only: the signed version, -mean(R over list) / SD(R over the TF's universe), reported per TF and pooled without a test, to show direction (activation vs repression). No other change. Status remains DECLARED.
