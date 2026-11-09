# Regulome gene layer, manifest v8 (frozen 2026-11-09)

Gene layer: **binding-only over ABC links** = B6 list (e) from regulatome/m4_validation/b6_core/b6_lists.py, top 1,000 genes per TF, all 79 panel TFs (lists/<TF>.tsv: gene, rank). Built by build_v8.py from b6_core/lists after verifying manifest_b6.sha256. Rule: per TF, sum of the TF's raw Gate 1 CPM over every element ABC-linked to the gene (m2_abc EnhancerPredictionsFull_thr0.02), no ABC weighting.

Methods line: **benchmark-tuned on 52 TFs, not confirmed on 24 held out.**

- win100k window variant not adopted (held-out FAIL).
- held-out set (split_v1.tsv, 24 TFs) is now SPENT and must not be reused for any later question.
- Rules and record: regulatome/m6_improve/SPEC.md (sha256 596a05b206d6adedaa2606501ac8530513e90be23d4e78ae7d3947dca36bcd40, Addenda 1-2), regulatome/m6_improve/SPEC_NETWORK.md (sha256 b00e67969f38fdbb9ef4451daa68fe25a5cf8783a2a4f0d49b80061c1e0c59c9), held-out commits 20fde83 (table) and 46bae6f (companion), write-up regulatome/m6_improve/heldout_result.md.

manifest_v8.sha256 is written last and covers every file in this folder except itself.
