# Fig 3 figure legend (draft, panel letters of the 2026-11-11 plan)

**Figure 3 | TF coordinated binding modules sit on Hi-C loops and assemble a super-enhancer hub.**

**a**, Schematic of higher-order coordinated binding for one TF. Top: single-cell HITAG binding at a promoter and two enhancers; coordination across cells (elements bound at the same frequency can be bound independently or together; only coordinated binding exceeds chance; Cicero pair score, k = 50); a module is three or more elements with every pair coordinated, edges over 25 kb; module edges tested against Hi-C loops, drawn as a rosette hub. Bottom: the pipeline, run separately for each TF: HITAG profiles (79 TFs, one library per TF, single HEK293T cells); MACS2 peaks per TF (q ≤ 0.05; median ~5,200 peaks); Cicero aggregates of 50 nearest-neighbour cells; pair scores per peak pair, pairs >25 kb; modules of ≥3 elements with every pair coordinated; Hi-C support (wild-type HEK293T Hi-C, Peakachu loops ≥0.7, both ends within ±30 kb of anchors); null of 10,000 shuffles preserving chromosome and spacing, BH q per TF.

**b**, RBBP4 example. Left: RBBP4 coordinated binding network (components ≥10 regions); coloured edges = higher-order coordinated binding modules, pink = module edge on a wild-type HEK293T Hi-C loop (Peakachu 5 kb ∪ 10 kb, score ≥0.7, ±30 kb), gold = module with every edge on a loop (15). Right: genome-wide circos of RBBP4 module edges with the Peakachu loops underneath (359 modules, 1,072 coordinated binding edges, 328 on a loop, 27 modules with every edge on a loop).

**c**, RBBP4 coordinated binding pairs >25 kb along every chromosome (above; blue, n = 19,524), with module edges (pairs inside higher-order coordinated binding modules) in dark purple (n = 1,031). Below: only the wild-type HEK293T Peakachu loops (score ≥0.7, 5 kb + 10 kb) that support a shown pair with both anchors within ±30 kb: 4,300 of 130,668 loops; dark red = supports a module edge (277), light red = supports other pairs only (4,023). 5,188 of 19,524 pairs (26.6%) sit on a loop.

**d**, Higher-order coordinated binding modules per TF with ≥1 edge >25 kb, stacked by the share of module edges on a loop; % = modules with ≥1 loop edge (ETS2, ZNF282 cut at 2,500). Right: % of module edges >25 kb on a loop (dot) against each TF's own null of 10,000 same-chromosome shuffles that move every edge independently, keeping its spacing (box = IQR, tick = median, whisker = 95%); filled = BH q < 0.05 (36 of 40 TFs with ≥100 module edges; median 21.7% vs 15.0%), open = n.s., n/a = <100 edges.

**e**, Genomic class (RefSeq Select promoter, TSS ±1 kb; otherwise HEK293T ChromHMM) of all Cicero peaks, all module regions and regions in modules of 1, 2, 3–4, 5–9 and ≥10 TFs.

**f**, TFs (left) and the 10 most widely shared module edges >25 kb (right, one per locus); a ribbon = that TF has this exact edge in one of its modules; TF shade = how many of the 10 edges it has; gene names only at TSS ±1 kb.

**g**, The HEK293T super-enhancer SE_02_195100173 (SEdb; chr19:1,248,039–1,254,347, gold) and its partners (bands: light blue promoter, light orange enhancer, grey other): Hi-C (10 kb), Peakachu loops, Cicero edges coloured by what they connect (solid = on a Hi-C loop, dashed = Cicero only), raw single-cell TF coverage, ChromHMM, RefSeq genes. 135 edges, 50 on loops; SE loop-edges contact the SE 1.86× more than matched pairs (p = 0.001; selected on loops, partly expected), other partner pairs 0.99×.

**h**, Most-closed structure of the Hi-C-constrained polymer model of the SE window (#9102; 14 of 18 SE loop-edges closed, top 0.07%; held-out ρ 0.14): an illustration; small spheres = TF peaks (the model has no TF input).

**i**, All 135 Cicero edges of the hub pooled across TFs and cells: SE in the centre, partners clockwise in genomic order from chr19:0.51 Mb, partner–partner edges drawn outside the ring. Legend for g–i as shown.

**j**, Biological processes of genes that modules sit on (promoters in modules of ≥3 TFs, 1,035 genes) vs all network promoter genes (filled; fold, BH q), and after matching for local peak density (hollow; none q < 0.05). Heat-response terms (BAG3, HSPA6, HSPB1, HSPH1; modules of ≥10 TFs) omitted: heat-shock genes with exon reads.
