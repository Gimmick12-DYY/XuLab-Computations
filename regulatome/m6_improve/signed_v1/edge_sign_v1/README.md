# edge_sign_v1: per-edge sign layer on v8 (SPEC_SIGNED.md sources B + C)
Rule declared 2026-11-10, committed BEFORE any signed knockdown z is read for this layer. Script: build_edge_sign.py.

Edges: TF x ABC-linked element x gene, for each TF's r6_v8 genes (top 1,000), over the m2_abc power-law links (EnhancerPredictionsFull_thr0.02); elements = M0 v2 active set (promoter, distal_active, bivalent).
Element state (source B): HEK293 18-state ChromHMM segmentation as used for the NuRD SpQN result (tf_complex/work/chromhmm_elements_counts/elements.tsv), state with the largest bp overlap per element, collapsed to:
active = TssA, TssFlnk, TssFlnkU, TssFlnkD, EnhA1, EnhA2, EnhG1, EnhG2, EnhWk, Tx, TxWk; pcg = ReprPC, ReprPCWk, TssBiv, EnhBiv; het = Het, ZNF/Rpts; quiescent = Quies or no overlap.
Co-binding (source C): repressive markers in the panel = EZH2, MTF2, MBD3, RBBP4, RBBP7, HP1 (SUZ12 not in panel; CBX1/3/5 are HP1); activating set = NRF1, MAZ, NFYA, E2F3. A marker flags an element if its raw CPM there is > 0 and >= its own 90th percentile over the active elements (each marker's own distribution, so depth-matched). The edge's own TF never counts as its own marker. rep_cobound / act_cobound = any marker of that set flags the element.
Edge sign: repressive context = state in {pcg, het} or rep_cobound.
- repressing if repressive context and not act_cobound;
- activating if (state active or act_cobound) and not repressive context;
- otherwise the TF baseline: canonical tf_sign_annotation.tsv direction if confidence high/medium (activator -> activating, repressor -> repressing), SREBF2 activating (source=rc79, high); dual, unknown or low confidence -> unsigned.
Gene sign (per TF, gene): binding-weighted majority, sum over the TF's linked elements for the gene of (TF CPM at element) x (+1 activating / -1 repressing / 0 unsigned); sign of the sum; zero -> unsigned.
Outputs: edge_sign_v1.tsv (TF, gene, element, state, flags, edge_sign, gene_sign), tf_departure_summary.tsv (per TF: fraction of edges / genes whose sign departs from the TF baseline).

Descriptive check (DESCRIPTIVE ONLY; held-out set spent; labelled dev/heldout from split_v1): per TF and arm, with z loaded as b6/m6 load() (>= 10% NTC universe, units averaged per TF), double-centred:
- concordance = mean over the TF's signed v8 genes of s_g x (-z_g) (s = +1 activating, -1 repressing; positive = direction matches) and act_minus_rep = mean z of model-activated minus model-repressed genes (expected < 0);
- compared with (i) the source A constant sign (every gene gets the TF baseline sign; only for TFs with a scored baseline) and (ii) 1,000 random sign assignments that keep each TF's numbers of activating and repressing genes (per-TF marginals), seed 20261109.
- Pooled across TFs and reported separately for TFs whose baseline is activator, repressor, and unsigned, as well as pooled, because the scored baseline set is repressor-heavy (a repressive-leaning model can look good for the wrong reason). Every pooled number with and without NRF1. Scoring uses high/medium-confidence baselines only (signed_v1/README.md).
