# Interface audit: what does "strong" mean? (2026-11-10)

Prompted by DDIT3–RBBP7 scoring strong in 5/5 AF3 seeds with only 2 confident DDIT3 interface residues. CPU only. This README is
committed **before** any audit output exists. All outputs are DESCRIPTIVE except the floor relabel, which follows the rule below.

## Metric
Confident interface residues (CIR): residues on EITHER chain with any heavy atom within 5 Å of any heavy atom of the other chain AND
residue pLDDT ≥ 70 (residue pLDDT = mean B-factor of its atoms; AF3 and Boltz-2 both write pLDDT 0–100 there). Reported per chain
(CIR_a, CIR_b) and in total (CIR = CIR_a + CIR_b). Two-chain protein models only; a model with more than two protein chains (e.g. the
EZH2–SUZ12–MTF2 trimer) is skipped and logged.
- 1-seed runs: the top-ranked model (`<pair>_model.cif`, the AF3 top; Boltz `screen/structures/<pair>.cif`, the top sample).
- multi-seed runs: per seed the top-ranked sample (max ranking_score in ranking_scores.csv); report per-seed CIR and the median across
  seeds. The median is the value used for the floor and for relabelling.

## Sets
1. AF3 1-seed sweep `af3_corr060_20261104/scores_s1` — every pair is measured (cheap), so deciles cover all of strong/weak/none.
   The declared comparators are a random sample of 100 weak and 100 none (numpy seed 20261110), flagged `comparator_sample`.
2. AF3 5-seed `af3_corr060_20261104/scores_s5` — every pair scored at run time (strong = n_seeds_strong ≥ 3, as in the rescore).
3. AF3 earlier runs: `af3_local/af3_scores.tsv` (includes the af3_inputs full-length 14-hit set), `af3_novel_20261103/af3_scores.tsv`,
   `af3_scores_weak.tsv`, `af3_scores_srcap.tsv` — every row with a model folder.
4. Boltz-2 full screen `screen/scores.tsv`, set != homo: every strong and weak row with a structure (non-hits have no saved model).
   Panel-level count (HP1 paralogs collapsed to the best by ipSAE, ipTM) is reported next to the row count.
5. Controls. Positive: EZH2–SUZ12 (AF3 af3_local and out_control), MTF2–RBBP4 (known PRC2; AF3 af3_local, Boltz), RBBP4–ZIC2 (known
   NuRD; AF3 sweep). Negative: RBBP7–SRCAP (AF3 af3_local / af3_novel). Every available model of a control counts as one observation.

## Floor rule (declared now, applied after the distribution is read)
- p10 = 10th percentile of total CIR over the positive-control observations (AF3 and Boltz pooled; multi-seed runs contribute their
  median). If fewer than 5 positive-control observations exist, controls are "too few" and the floor is 10.
- Floor = min(p10, 10).
- A strong call with CIR below the floor is relabelled `strong_noCIR` and excluded from strong counts. Weak/none labels are unchanged.
- Report: how many AF3 and Boltz strong calls move; which rows of the deck pair table
  (`triple_support_20261109/triple_support.tsv`: AF3_only / T1–T3) and which 5-seed strong calls change.

## Also reported
Per tool, CIR deciles for strong / weak / none; Spearman and Pearson correlation of CIR with ipSAE (max) and ipTM (top).
Where DDIT3–RBBP7 sits: its CIR, its percentile among AF3 strong calls and among positive controls.

## Outputs
`audit.tsv` (one row per pair × source, with per-seed CIR), `summary.tsv` (deciles, correlations, floor, moves), `moves.tsv`.
Not in scope (not queued): DDIT3 docking against random panel TFs.

## Amendment 1 (2026-11-10, before regenerating tables)
(1) The total-CIR floor is RETIRED. It does not discriminate: the negative control RBBP7–SRCAP scores 106, above the positive-control
p10 of 76.5. The strong_noCIR relabel is withdrawn for every row.
(2) Min-chain CIR ≥ 5 is a FLAG, not a gate. Rows below it keep their call and carry the footnote 'one-sided interface, peptide-scale
(N confident residues on TF X)'. The threshold 5 is fixed and must not be tuned. Rationale: RBBP4–ZIC2, a published NuRD pair, has 8 on
one side.
(3) Boltz: a column 'boltz_only_no_interface_support' marks every row whose only structural support is Boltz. The reason is that Boltz
CIR is unrelated to its own scores (ipSAE −.10, ipTM −.24; median CIR strong 38 vs weak 41). Whether to keep these rows is Yuyang's
decision; nothing is removed here.

Operational definitions (fixed with this amendment, before any flag table exists):
- Min-chain CIR = min(CIR_a, CIR_b) of the model(s) behind the call (multi-seed: per-chain medians across seeds, as in audit.tsv);
  X = the chain with the smaller value (ties: both named).
- flags.tsv: every strong row of audit.tsv (AF3 and Boltz). Footnote when min-chain < 5. boltz_only = the pair (HP1 paralogs collapsed)
  has a Boltz strong row and no AF3 strong row in any AF3 run in audit.tsv.
- triple_support_flagged.tsv (copy of triple_support_20261109/triple_support.tsv; original untouched): the min-chain value comes from the
  AF3 run named in af3_source (sweep_s1 → af3_s1_sweep; af3_local / af3_novel_20261103 → that run's rows, max over its score files), and
  from af3_s5 when af3_s5_tier is strong. The footnote is attached when the row's AF3 call (af3_tier or af3_s5_tier) is strong and its
  min-chain < 5; for a Boltz-only row, from the Boltz model. boltz_only_no_interface_support = boltz_trimmed or boltz_fulllength is strong
  and neither af3_tier nor af3_s5_tier is strong.
