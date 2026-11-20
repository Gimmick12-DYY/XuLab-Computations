# Round 7: alternative own-TF agreement tests (declared 2026-11-20, before any result)

Raw single-cell reads only (5' ends, inputs/five). Imputed is excluded until the raw-vs-imputed label reconciliation
(r18b_imputed_reconcile.py) is reported. Panel = CTCF, MAZ, ZBTB7A, ZIC2, ZNF282, ZNF777. References as in fig2_common
(CTCF = ENCODE HEK293 ENCFF314ZAL summits; ZNF282 = GSM2466515 HEK293T; others ENCODE HEK293). Pool = all 79 TFs; every
"pool" below is leave-one-out (minus the scored TF).

1. **Pairwise discrimination AUROC.** For each reference pair A,B (A != B): A-unique = A summits whose summit is more than
   500 bp from every B peak; B-unique likewise; top 2,000 of each by signalValue. Score of sc track X at a summit =
   log2((X reads +-250 bp + 0.5) / (pool reads +-250 bp + 0.5)). Cell (X, A) = mean over the 5 B of AUROC(A-unique vs
   B-unique) using X's score. Diagonal = own. Depth: full and thinned to ZNF777 depth (420,339 reads; seeds 11, 22, 33,
   AUROC averaged). Halves: odd vs even + X chromosomes (summits split by chromosome).
2. **Motif-centred profiles.** Reference-unique peaks (inputs/unique_<TF>.bed, not overlapping any other panel reference).
   Motif: HOCOMOCO v11 CTCF_HUMAN.H11MO.0.A, MAZ_HUMAN.H11MO.0.A, ZBT7A_HUMAN.H11MO.0.A, ZIC2_HUMAN.H11MO.0.D,
   ZN282_HUMAN.H11MO.0.D; ZNF777 has no HOCOMOCO motif, so its motif = STREME motif 1 from its top 500 unique summits
   +-50 bp vs shuffled control. Best FIMO hit (p < 1e-5) within summit +-100 bp; anchor = motif centre, oriented by strand.
   Profile = 5' ends +-500 bp, 10 bp bins; centre/flank = mean within +-50 bp / mean at 300-500 bp both sides.
   All rows thinned to ZNF777 depth (3 seeds averaged). Cell value = z of log2 centre/flank vs 50 pseudo-TFs (random
   420,339-read draws from the 79-TF pool) at the same sites.
3. **FRiP-style enrichment.** Cell (X, A) = log2[(fraction of X reads in A-unique peaks) / (same fraction for the pool)],
   full depth; halves as in 1.

Statistics for every 6x6: diagonal minus off-diagonal, exact 720-permutation p of row labels; the same on the 5x5 without
CTCF (120 permutations); own-best rows/columns; each half separately. A visible diagonal = own cell the row maximum.
Outputs results_293T_analysis/fig2_qc/round7_alt/; Mac figures_fig2_qc/round7_alt/.
