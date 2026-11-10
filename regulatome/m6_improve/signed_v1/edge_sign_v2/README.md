# edge_sign_v2: declared fix for v1's saturated co-binding flags (DESCRIPTIVE)
Rule declared 2026-11-10 ~00:55 local, committed BEFORE any v2 output exists or is read. Script: build_edge_sign_v2.py (built from ../edge_sign_v1/build_edge_sign.py).

Only change from v1 (../edge_sign_v1/README.md + Amendment 1): the co-binding flag.
- Enrichment of marker m at element e over the element's pooled panel signal: E_me = log2((CPM_me + 0.5) / (pooled_e + 0.5)), pooled_e = mean CPM over the 79 panel TFs at e.
- For an edge of TF t, pooled_e excludes t: pooled_e(t) = (sum over 79 TFs - CPM_te) / 78. The marker's own CPM stays in the pool.
- Flag fires if E_me >= the marker's own 90th percentile of E over the active elements (percentile computed once per marker with the full 79-TF pool) AND CPM_me >= the marker's median CPM over the active elements.
- The edge's own TF is excluded from the marker sets (repressive: EZH2, MTF2, MBD3, RBBP4, RBBP7, HP1; activating: NRF1, MAZ, NFYA, E2F3).
Everything else identical to v1: ChromHMM state collapse, edge sign rule, binding-weighted gene majority, high/medium canonical baseline only (SREBF2 from rc79), class split on every row, pooled with/without NRF1 and with/without the 10 marker TFs, random-marginal (1,000 draws, seed 20261109) and constant-sign comparators, z as b6/m6 load(), double-centred.

Pre-declared sanity gate: flag rates over all edges are computed and written to flag_rates.tsv first. If either the repressive or the activating co-binding flag fires on > 50% or < 2% of edges, the script stops before reading any knockdown z (writes GATE_FAILED.txt) and the result is reported without scoring.
