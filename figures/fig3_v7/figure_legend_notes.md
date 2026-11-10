# Fig 3 figure legend (draft text, v7)

**Figure 3 | TF co-binding modules sit on Hi-C loops and assemble a super-enhancer hub.**

**a**, Higher-order co-binding modules per TF with ≥1 edge >25 kb, stacked by the share of module edges on a wild-type HEK293T Hi-C loop (Peakachu 5 kb ∪ 10 kb, score ≥0.7, ±30 kb); numbers = % of modules with ≥1 loop edge (ETS2 and ZNF282 bars cut at 2,500, //). Right: % of module edges >25 kb on a loop (dot) against each TF's own null of 10,000 same-chromosome shuffles that move every edge independently and keep its spacing (box = IQR, tick = median, whisker = 95% interval); filled = BH q < 0.05 (36 of 40 TFs with ≥100 module edges; median 21.7% observed vs 15.0% null); open = n.s. (SMARCA1, FLYWCH1, HOXA10, ZNF512); n/a = <100 module edges. TFs with ≥20 modules shown.

**b**, Genomic class of regions (RefSeq Select promoter = TSS ±1 kb; others by HEK293T 18-state ChromHMM at the region midpoint) for all Cicero peaks, all regions in higher-order co-binding modules, and regions in modules of 1, 2, 3–4, 5–9 and ≥10 TFs.

**c**, The HEK293T super-enhancer SE_02_195100173 (SEdb; chr19:1,248,039–1,254,347, yellow band) and its partners (grey bands). Tracks: wild-type Hi-C (10 kb, ICE-balanced), Peakachu loops (0.7), Cicero co-binding edges coloured by what they connect (solid = on a Hi-C loop, dashed = Cicero only), raw single-cell TF read coverage (CPM, own scale per track), ChromHMM, RefSeq Select genes. The hub has 135 Cicero edges, 50 on Hi-C loops: from the SE 7 + 5 to promoters, 5 + 1 to enhancers and 6 + 7 to other sites (loop + Cicero only); between partners 8 + 8 promoter–promoter, 16 + 26 involving an enhancer and 8 + 38 other. The 18 SE loop-edges contact the SE 1.86× more than distance- and TF-class-matched pairs (1,000 sets, p = 0.001; edges were selected on loops, so enrichment is partly expected); other partner pairs as a set 0.99× (p = 0.67).

**d**, The most-closed structure of the Hi-C-constrained polymer model of the SE window (structure #9102; 14 of 18 SE loop-edges closed, top 0.07% of 10,000 structures, typical = 7; held-out ρ 0.14): an illustration, not proof of a rosette. 23 of 31 partners are linked into one core; small spheres = TF MACS2 peaks at each element (the model has no TF input).

**e**, All 135 Cicero edges of the hub, pooled across TFs and cells (not one simultaneous structure): the SE in the centre, partners on a ring clockwise in genomic order from chr19:0.51 Mb; partner–partner edges drawn outside the ring.

Legend for c–e as on the figure. Supplementary: pairs with Hi-C loop support and their per-pair null (all 41 TFs q < 0.05), the stricter whole-module block-shuffle null (35 of 40 TFs), the RBBP4 co-binding network, the SE edge-count table.
