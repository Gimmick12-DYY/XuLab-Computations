# Research Journal — XuLab-Computations

A durable, chronological record of **major** updates: what changed, why, and the
outcome/decision. This is the place to review the arc of the project later.

**Scope.** Single-cell TF binding for the ~78-TF HEK293T scCUT&Tag "TF1000cells"
panel: imputation of sparse per-TF binding, benchmarking against bulk, and
downstream biology (peak calling, motif enrichment, TF co-occupancy / complexes,
A/B compartments, ChromHMM state analysis).

**How to use this file.**
- Newest entries at the top of the log (reverse-chronological).
- One entry per *major* update (a new pipeline, a resolved confound, a corrected
  method, a key biological finding). Routine commits stay in git.
- Each entry: date, one-line title, **Context / Change / Outcome** (+ **Files** or
  **Commit** when useful). Convert relative dates to absolute.
- Seeded 2026-09-13 from git history, the memory notes, and
  `CHANGES_July_to_August_2026.md` (which remains the detailed July→mid-Aug record).

**Environment note.** Code runs on the Longleaf cluster
(`/work/users/d/y/dyy12/XuLab`) via commit → push → pull; a stale checkout on the
cluster is the usual cause of "unchanged output."

---

## Log (newest first)

### 2026-11-10 — Full-matrix, sign and edge-sign checks all null; testability diagnostic; no paired RNA
- **Context:** After the held-out FAIL and v8 freeze (2026-11-09), Yuyang asked for a full-matrix comparison and a signed (activator/repressor) Regulome. All checks below are DESCRIPTIVE (held-out set spent).
- **Change:** regulatome/m6_improve/: SPEC_MATRIX.md, SPEC_SIGNED.md; matrix_v1/ (matrix_compare.py, matrix_mispaired.py; Correction 1 declared before output read: mis-paired S as the double-centred primary control, uncentred element count secondary); signed_v1/ (canonical source A sign table + readout write-up, machine-built rc79 table for comparison, annotation comparison, high/medium-only scoring rule); signed_v1/edge_sign_v1/ (per-edge sign from ChromHMM state + repressive/activating co-binding, rule and Amendment 1 committed before any signed z was read); validation_programme_summary.md.
- **Outcome:** (1) matrix_v1: per-TF Spearman of our score vs knockdown z over all linked genes is ~0 (median |rho| < 0.005) in every variant; a TF's own score does not beat another TF's (mis-paired) score after double-centring; S_v8 = S_abc (paired difference ~0.001, CI spans 0). (2) Sign annotation vs knockdown direction agrees at chance (25/45; an independent readout over v8 targets gives 24/45, 10/22 high-confidence); 39/76 TFs uncalled. (3) Edge-sign layer v1: no gain over the constant per-TF sign or over random signs with the same marginals (Arm 2 model − random +0.002 [−0.008, 0.012]); the 90th-percentile co-binding flags hit ~80% of v8 edges for both marker sets, so edges mostly inherit the TF baseline. (4) Testability diagnostic: only E4F1, NME2 and XPA are testable on both axes (binding-list split-half Jaccard median 0.107, knockdown ceiling median 0.10, r = +0.04 between them). (5) No paired RNA exists for the HITAG data on Longleaf (DNA libraries only; Ren-lab pipeline paths suggest any RNA arm would be upstream at UCSD; Yuyang confirmed none).
- **Files:** regulatome/m6_improve/{SPEC_MATRIX.md, SPEC_SIGNED.md, matrix_v1/, signed_v1/, validation_programme_summary.md}.
- **Addendum (2026-11-10):** Edge-sign layer v2/v3: v2 stopped at its pre-declared flag-rate gate (54.7%) before any knockdown data were read. v3 (stricter flag, declared 5e176d0) is informative but null against random signs, and for Arm 1 activators it is worse than the constant TF-class sign (−0.040, CI excluding 0). Reason: 81% of elements are in active states, so state plus co-binding approaches a constant 'active' call that overrides TF identity (repressor-baseline TFs receive 74% activating genes). Finding: element state cannot override what the TF is; direction belongs to the protein, not to the neighbourhood it sits in. Both of today's ideas, network borrowing and sign from element context, were tested and answered.

### 2026-11-09 — Regulome gene layer tuned against Perturb-seq; held-out FAIL; manifest v8 frozen
- **Context:** B6 (third declared Perturb-seq read) showed the binding × ABC core beats accessibility-only but loses to binding-only on Arm 2. Yuyang lifted the validation-only rule: the HEK293T X-Atlas/Orion knockdowns become a development benchmark. B7 (HCT116 confirmation) was staged but dropped before any download or read.
- **Change:** regulatome/m6_improve/. TF-level split split_v1.tsv (52 development / 24 held-out, ceiling-stratified, NRF1 in development), sealed before any grid result. Grids 1–6 on development TFs only; objective changed after grid 2 to ceiling-weighted own-vs-other-KD specificity, with element-count, pooled other-TF and ATAC controls (SPEC.md); learned model (model1) and element-level pooled normalisation (grid5) tried. Network borrowing across TFs (SPEC_NETWORK.md, net1b): kNN5 / dense co-binding + complex edges vs uniform pooling and degree-preserving swap, fixed and reliability α.
- **Outcome:** ABC weighting is harmful to the gene score; ATAC × ABC scores below random genes; a plain ±100 kb distance window matches the linked-element machinery (dev lead over ABC-sum about +0.4, not over element count). The raw B6 effect is mostly generic responsiveness (no-binding controls score as high). Network borrowing: co-binding and complex edges carry no detectable shared knockdown response vs uniform pooling and the degree-preserving shuffle; weak TFs do not gain. Held-out single scoring (job 8702375, commits 20fde83, 46bae6f): FAIL, the pre-declared null — win100k Arm 2 Wspec −0.14 [−0.75, 0.51], margin over element count −0.34 [−0.88, 0.20]; controls score higher. Frozen as manifest v8 (regulatome/r6_v8/): binding-only over ABC links (B6 list e), 79 TFs; window variant not adopted; held-out set spent. Methods: benchmark-tuned on 52 TFs, not confirmed on 24 held out.
- **Files:** regulatome/m6_improve/ (SPEC.md, SPEC_NETWORK.md, split_v1.tsv, grid1–6, model1, net1b, heldout_v1, heldout_result.md), regulatome/r6_v8/.

### 2026-11-06 — TF groups by shared co-binding pairs/cliques; clique location
- **Context:** Ask whether groups of TFs share Cicero pairs/cliques beyond depth, and where cliques sit.
- **Change:** cobinding/tf_groups/: degree-preserving (curveball) null keeping TF depth, pair popularity and library
  stratum; Leiden groups; STRING/CAP-SELEX/ChromHMM/GO tests vs depth-matched random groups; clique-vs-node location
  with local-density matching; WT Hi-C compartments at 100 kb; GO files in data/go.
- **Result:** Sharing structure = library quality + depth; core vanishes at top-1,000 depth matching; no STRING/CAP-SELEX
  signal; clique chromatin preferences mostly explained by local node density. See cobinding/tf_groups/README.md.

### 2026-11-02 — AF3 check of the Boltz hits; R1 TF-TF network frozen; R2 targets propagated
- **Context:** the Boltz-2 screen (below) gave 71 strong pairs, but several were construct- or
  seed-dependent. Before freezing the TF-TF layer, the 38 Boltz hits worth keeping were re-folded
  with the Longleaf AlphaFold3 install (module `container/alphafold/3.0.1`, UNC central weights,
  full-length UniProt canonical sequences, 5 seeds).
- **Change:** `complex_screen/af3_local/`. Controls first: EZH2-SUZ12 strong 5/5 seeds (ipTM 0.71,
  ipSAE 0.77); RBBP7-SRCAP none (needed a 128 GB rerun). Then the 38 pairs (20 strong/changed +
  18 complex_weak).
- **Outcome:** AF3 strong 4, weak 8, none 26 → Boltz over-calls. Strong: MLX-SREBF2 (ipTM 0.45,
  ipSAE 0.57, 5/5; the only pair strong in Boltz trimmed + Boltz full-length + AF3), NEUROD4-SREBF2
  (NEUROD4 not expressed), HOXC4-RBBP4 (3/5), MTF2-RBBP4 (1/5). AF3 supports none of the complex
  edges (DR1-THAP6 none despite Boltz full-length strong 3/3; RBBP4-RBBP7 none, paralogs).
  EZH2-MTF2 dimer: no contact (ipSAE 0); EZH2-SUZ12-MTF2 trimer: SUZ12-MTF2 0.42/0.39 5/5, EZH2-MTF2
  0/5, i.e. bridged by SUZ12. Superposition checks: MLX-SREBF2 on MYC-MAX (PDB 1NKP) Cα RMSD 0.95 Å
  over 110 residues (supports the bHLH-ZIP fold, not the partnership); MTF2 on SUZ12 sits on a
  different surface from its paralog PHF19 in PDB 6NQ3 (2/20 contact residues shared) → AF3
  interface likely wrong despite confidence.
- **Freeze rulings:** Boltz-only/trimmed-only edges → weight 0; AF3-strong MLX-SREBF2, HOXC4-RBBP4,
  MTF2-RBBP4 = physical-only w 0.5; NEUROD4 edges w 0 (not expressed); DR1-THAP6 → complex_weak
  (af3_none); DR1-HOXC9 → co-localized; Boltz "none" on known pairs = no information.
- **Frozen R1** (`regulatome/r1_tftf/r1_final_classes.py --freeze` → `r1_tftf/v2_final/`,
  `frozen_manifest.sha256`, 205 files, sha256 `735357ff…593f`): complex 4 (EZH2-MTF2, WIZ-ZNF644,
  PCGF6-RBBP4, RBBP4-RBBP7), DNA-guided 1 (FOXK1-MLX), complex_weak 20, physical-only structure 60,
  physical-only DNA 2, physical-only weak 97, co-localized 231, STAG2-wrap 11, none 2,655;
  30 pairs carry weight > 0.
- **Propagation** (`regulatome/r2_targets/propagation/`): 17/79 TFs gain targets (median 934 genes;
  sparse gainers CHAMP1 EPAS1 FOXK1 HOXC4 HOXC9 MSX2 PCGF6 SREBF2 THAP6). Locally supported 17%
  overall, but module members are circular by construction (22-35%) and pair-transferred lists
  only 7-9% vs ~5% chance.
- **After the freeze:** user asked whether this much cis-regulation is too little → a third
  transfer class "cis-coordinated" (Cicero excess + a second co-binding metric ≥ 0.90; loop-supported
  w 0.5, plain w 0.25) was approved for a re-freeze (manifest v2). The re-freeze is PARKED:
  `r2_targets/r2_cis_v2.py` fails because 61/79 TFs lack rows in the shared-edge matrix (treat as
  zero shared edges on restart). Ablation arms (promoter-only, imputed residual, blend) are built in
  `regulatome/m4_validation/arms/` but not hashed. v1 manifest untouched.
- **Rule kept:** X-Atlas/Orion Perturb-seq is used only to compare against the frozen output; no KD
  file has been read. Validation plan (stream 223 HF parquet batches, 74 testable TFs) waits for
  manifest v2.

### 2026-11-01 → 11-02 — Composite-motif test for complex pairs: not detectable at sc depth
- **Context:** PI asked for a novel-complex example supported by correlation + shared motif + AF3.
  Earlier layer-3 pilots were negative, so a rank-free "reverse" test was designed: define elements
  by sequence, then measure the pair's co-binding residual.
- **Change:** `complex_screen/composite_motif/`. Calibration on CTCF (union-ATAC elements with vs
  without MA0139.2). Pair test = composite (motif A + B ≤ 50 bp) vs single-motif elements within
  expected-read bins, bootstrap CI, pass needs product AND min residual. Pairs: FOXK1-MLX
  (pre-registered CAP-SELEX config), MLX-SREBF2, DR1-THAP6 (THAP1 proxy), WIZ-ZNF644; negative
  control RBBP7-SRCAP. Last exploratory run: trend of co-binding residual over composite-score
  deciles (`trend_exploratory/`).
- **Outcome:** CTCF fails the pre-specified 1e-4 test; passes only post hoc with read-bin matching
  and dose response (1e-5 1.07×, 1e-6 1.16×, 1e-7 1.20×), and shows no monotone trend. No pair
  passes: MLX-SREBF2 only 15 composite elements (product 0.80, CI 0.37-1.21); FOXK1-MLX 0 elements;
  THAP1 proxy 0 hits; WIZ/ZNF644 have no usable PWM (Schmitges 2016 / CIS-BP unreachable). The
  negative control "passes" at the same 15 elements → small-n CIs untrustworthy.
- **Decision:** motif work stopped. Our reads give an honest negative (low power, does not rule
  out complexes); the motif leg for the PI cites CAP-SELEX / literature. EZH2-MTF2 CpG test
  (`complex_screen/jie_motif/`, MTF2 EH domain binds CpG): co-bound CpG o/e 0.46 = matched, ratio
  0.99 → negative.

### 2026-11-01 → 11-02 — PI requests: complex count table, worked complexes, Hi-C clique triangle
- **Counts** (`complex_screen/complex_counts/`, normalized all-elements ChromHMM SpQN percentile):
  known 13 (STRING/CORUM + literature pairs ZNF512B-RBBP4, ChAHP, WIZ-ZNF644), 3 recovered at
  pct ≥ 0.90 (MAZ-ZNF143 0.97, EZH2-MTF2 0.96, WIZ-ZNF644 0.94); recently revealed = CAP-SELEX
  pairs only (FOXK1-MLX, MLX-MSX2, HOXD10-SREBF2); novel AF3-supported 2 (MLX-SREBF2, HOXC4-RBBP4).
  NuRD fails on the corrected matrix (0.50): the old "NuRD confirmed" came from RBBP4-RBBP7
  paralog correlation (MBD3 pairs 0.34-0.66).
- **Examples:** known = EZH2-MTF2 (corrected pct enh/repr/Polycomb 0.93/0.94/0.96, within local
  cluster 0.77/0.84/0.90); novel = MLX-SREBF2 (0.78/0.44/0.81; local 0.56/0.44/0.73; the raw
  repressive signal was accessibility). MLX-SREBF2 was tested in CAP-SELEX and NOT called; no
  co-binding edge. Raw Wang-style ChromHMM correlations are > 0.9 for nearly all pairs and their two
  blocks are exactly the 25/54 library split → only corrected values are reported.
- **Clique triangle** (`results_3d_model/clique_triangle/`, Mac): 1,903/49,823 cliques contain a
  loop-edge triangle; with ±20 kb anchor shift, min dot/donut ≥ 1.5 in 89 vs 13 null. Chosen ETS2
  clique on chr1 (ACBD6 promoter / distal / LINC01732), dot/donut 2.21/2.26/2.18 at 10 kb. 66% of
  triangles are ETS2 cliques.

### 2026-11-01 — R2 direct targets for all 79 TFs; Boltz full-length retests
- **Change:** `regulatome/r2_targets/`. S(q,g) = Σ_e residual binding × power-law ABC × (1 + κ·Cicero),
  κ = 0/1. Direct list = top 1,000 genes by S (κ = 1 primary). Propagation sources = the specific
  TFs plus module pools (NuRD = MBD3 only; WIZ/ZNF644 broad-domain group = DR1 THAP6 CHAMP1 WIZ
  ZNF644 SRCAP; PRC2 core = EZH2 MTF2 PCGF6, flagged not_PRC2_enriched). Every source drops genes
  in the ATAC-only top 1,000 (`sources_atacfiltered/`).
- **Outcome:** tiers 7 specific / 33 accessibility-limited / 39 undetermined. List size tracks depth
  (specific median 7,998 genes with S > 0; undetermined 348; SETDB1 1; ZNF503 0) → use ranks. Top-500
  overlap with the ATAC-only list 4-33 (chance ~13) vs 114-205 uncorrected. κ barely matters
  (κ0/κ1 top-1,000 share 998-1,000). ZNF777 capped to 202 genes (shallow_source).
- **Bug:** RBBP4's Cicero output lives in `RBBP4_sweep_k50_i10000_s555`; fixed in R2. The Gate 1 v2
  RBBP4 raw+Cicero arm was probably skipped (noted in `gate1/v2_1/README.md`; ABC tier unaffected).
- **Boltz full-length retest of 6 trimmed-only complex edges (seeds 1-3):** DR1-THAP6 strong 3/3;
  DDIT3-XPA and DR1-HOXC9 seed-unstable; CBX1-ZNF507, DR1-ZNF746, ZNF143-ZNF512 lost.

### 2026-10-30 — Binned Gate 1, PRC2 library checks, repressive matrices: no Polycomb-shaped PRC2
- **Context:** ATAC-based ABC units discard reads in closed domains, so repressors (EZH2: 67
  scoreable genes) could be invisible at gene level.
- **Gate 1 v3 binned** (`regulatome/gate1/v3_binned/`; 10 kb, 50 kb, ReprPC domains, gene body
  ±10 kb; expected from panel + GC + mappability, no ATAC): binning does not expand the tier.
  Passes 10 kb 7 (E4F1 MBD3 MLX NFYA NME2 RBBP4 SOX4), 50 kb / gene body 3 (E4F1 RBBP4 SOX4), ReprPC 3
  (E4F1 MLX RBBP4); all ≥ 12k cells. Robust core RBBP4, E4F1; ZNF507 and ZNF777 fail every unit.
- **EZH2:** not shallow (3,970 cells, 258 reads/cell) but reproducible-not-specific; reads 10.0%
  active / 3.9% ReprPC / 86.1% other = panel median. ENCODE SUZ12 HEK293T peaks (ENCSR090ZDO):
  EZH2 rank 60/79, MTF2 50, PCGF6 57 (RBBP4 8, RBBP7 4). HOX clusters: PRC2 libraries 0.75-0.88× panel
  (`regulatome/hox_check/`). → PRC2 libraries carry a weak uniform accessibility signal, no PRC2
  occupancy. **Decision (user):** PRC2 stays a complex node on the TF-TF layer; its gene edges are
  flagged "not_PRC2_enriched".
- **Modules:** WIZ/ZNF644 broad-domain group (DR1 THAP6 CHAMP1 WIZ ZNF644 SRCAP, ± EZH2/MTF2) ranks 1-3
  of 21 vs depth+library-matched random pools on every unit → real module (do not call it
  PRC2-adjacent). Low-depth repressive set (SETDB1 NEUROD4 SOX12 ZNF79 ZNF606 ZNF777 ZNF581) and ChAHP
  (ADNP/HP1/CHAMP1, `regulatome/chahp_test/`) are NOT modules; HP1/ADNP libraries mildly H3K9me3-enriched.
- **Repressive ChromHMM matrices** (`regulatome/repressive_chromhmm_matrix/`, corrected version
  `repressive_chromhmm_corrected/`, same pipeline as the enhancer-element matrices): PRC2 EZH2-MTF2
  recovered on repressive elements (0.91-0.97) but equally on enhancers → not repressive-specific;
  repressive-only block = 16 low-library TFs (artifact). Repressor-subset matrix in
  `regulatome/repressor_matrix/`.

### 2026-10-30 — Worked example: HOXB8-RBPJ fails; MAZ-ZNF143 chosen
- HOXB8-RBPJ: SpQN pct 0.21-0.27; Boltz contact via the disordered HOXB8 N-terminus (61/87 contact
  residues at pLDDT 39) → rejected. **Decision (user):** MAZ-ZNF143 is the worked example (Pearson
  0.99/0.97, SpQN 0.99/0.97, pair-level module pass; no Boltz interface); MLX-SREBF2 kept as
  "promising, unconfirmed".
- All-pairs distal co-binding (`complex_screen/pilot_pairs/cobind_allpairs/`): raw obs/exp is
  depth-confounded (rho -0.82 with set size; median pair 3.6×) → report depth-matched percentiles
  (MAZ-ZNF143 79th, MLX-SREBF2 52nd, CTCF-STAG2 98th, RBBP7-SRCAP 20th).

### 2026-10-29 — Gate 1 v2/v2.1: a TF-level tier of 7; PRC2 and NuRD as module nodes; layer 3 negative
- **Context:** Gate 1 v1 passes (36/79) included shallow TFs such as NEUROD4 (not expressed).
- **v2** (`gate1/v2/`; GC + mappability covariates, single-donor null): 0/79 pass. Two causes: the
  single-donor null measures other TFs' biology, and at near-zero counts log((obs+.5)/(exp+.5)) is a
  covariate function, so shallow profiles reproduce and resemble each other (ZNF503 best-other for
  60/79).
- **v2.1** (user choice; `gate1/v2_1/`): v1 reproducibility AND deep-TF specificity margin > 0,
  count-aware residual (expected ≥ 5 per half). **Tier (frozen for design): RBBP4, NFYA, SOX4,
  ZNF507, MLX, E4F1, ZNF777.** EZH2, MBD3, XPA out (underpowered). CTCF and MAZ fail. Caveat: the
  tier overlaps the old deep "promoter hub" set.
- **Modules:** with RBBP4/7 removed, PRC2 core (EZH2 MTF2 PCGF6) and NuRD (MBD3) are distinct
  (+0.056/+0.066, 20/20, CGI-independent) → two module nodes. MAZ-ZNF143, HP1/ChAHP, MEF2, cohesin,
  G9a-WIZ, HES1-RBPJ = TF-TF edges only. leiden_1 (HOX/ZIC/SREBF2 group) = low-library, provisional.
- **Library batch:** 25 TFs put ~24.5% of reads in elements vs 16-19% for the other 54, with identical
  class profiles; this matches ChromHMM "Block A" → Block A is a library/batch split, not a complex.
- **Layer 3 pilot** (`complex_screen/pilot_pairs/layer3_v2/`; residual-selected distal elements, GC ×
  CpG-matched background, 20 shuffles): CTCF motif 45.8% vs 43.2% background, below all shuffles; no
  partner motif or spacing in any pair. → motif layer NEGATIVE at sc depth; Fig 4 motif evidence must
  come from CAP-SELEX + bulk.
- **Figures:** palette Option C "Continuity" adopted for all paper figures
  (`/mnt/project-files/palette/regulome_palette.py`, `rp.use("C")`).

### 2026-10-27 → 11-01 — Boltz-2 structure screen of all 3,081 panel TF pairs
- **Context:** PI's three-layer complex search for Fig 4: structure → correlation → motif. No AF3
  weights at the time → Boltz-2 2.2.1 (`envs/boltz`, `XuLab/complex_screen/`).
- **Calibration** (`regulatome/ppi_pilot/`, `complex_screen/calib/`): 8 known complexes vs 7 random
  pairs, ipTM/ipSAE AUROC 1.00; strong = ipSAE ≥ 0.3 & ipTM ≥ 0.4, weak = ipSAE ≥ 0.1. DNA-bound check:
  MYC-MAX (1NKP) 5/5 correct (1.2 Å); HOXB13 dimer (8BYX) top model wrong (13.6 Å) at ipTM 0.97 →
  confidence does not catch DNA placement errors.
- **Screen:** 3,081 heterodimers (+ homodimers), disordered regions trimmed (AFDB pLDDT ≥ 50), chains
  ≤ ~1,000 aa; array 7099032, low-memory settings. Final: strong 71, weak 108, none 2,902.
- **Artifacts found:** STAG2 hits = "STAG2-wrap" (semi-disordered partners wrap the HEAT surface /
  CES pocket; own class, no weight). hub_partner flag (ZUP1, STAG2, RBBP4, SREBF2, MSX2, MLX, ZNF746,
  ZNF282, EZH2, ZNF367): 58/71 strong hits involve one. MTF2-RBBP4 = trimming artifact.
- **Full-length rerun of 14 strong non-STAG2 hits** (`complex_screen/fulllength_check/`): robust 2,
  changed interface 5 (MLX-SREBF2 stays strong), construct-dependent/lost 7.
- **CAP-SELEX** (Xie 2025, supp. tables in `complex_screen/cap_selex/`): 19/79 panel TFs screened,
  95 panel pairs tested, 3 called (FOXK1-MLX, MLX-MSX2, HOXD10-SREBF2), no composite motif among them.
  The 92 negatives are annotation only (DBD-only in vitro), never a null.
- **R1 co-binding view:** STRING physical pairs sit at median pct 0.42-0.62 (known complexes are not
  typically at the top); cross-library pairs are lost (1/17) → complex edges come from structure +
  STRING; co-binding alone gives the "co-localized" class.

### 2026-10-26 → 10-28 — Regulome: target set, universe, binding layer fails, ABC wiring
- **Context (user, 2026-10-26):** main target = a TF regulatory network, the **Regulome**, from
  complex co-binding, Cicero networks and motifs, validated against X-Atlas/Orion HEK293T CRISPRi
  Perturb-seq. **Hard rule (2026-10-28):** Perturb-seq is only a final comparison; it never builds,
  selects, tunes or filters the model. Motifs = validation layer only.
- **Phase 1** (`regulatome/phase1/`): 74/79 TFs have significant on-target KD (fail EPAS1, NEUROD4,
  SRCAP, ZBTB40, ZNF606); data are on FigShare (350 GB h5ad) and HF `Xaira-Therapeutics/X-Atlas-Orion`
  (223 parquet batches). Paused until the network is frozen.
- **M0 universe** (`regulatome/m0_universe/`, GENCODE v46 in `data/gencode_v46/`): 159,965 active
  elements (≥ 2 kb tiled to ~1 kb); `elements_v2.bed` adds repressive classes (396,014 elements,
  362.9 Mb).
- **M1 binding FAILS sanity** (`regulatome/m1_binding/`): raw obs/exp median AUROC 0.49 vs 7 bulk ChIP
  (ATAC alone 0.87); most TFs have < 1 read per element. Repressive classes are below background for
  all TFs; no PRC2→Polycomb or HP1/SETDB1→heterochromatin signal. → element-level binding cannot come
  from our reads for most TFs; TF-specific signal exists only in aggregate for deep/motif TFs.
- **M2 ABC** (`regulatome/m2_abc/`, ABC v1.1.2, ATAC-only activity, power-law contact primary; WT
  GSE216399 Hi-C arm supplementary): 104,071 links over 19,807 genes, median 4/gene, distance median
  14.8 kb; Hi-C arm Jaccard 0.93.
- **Design** (doc "Regulome unified model design"): two-resolution Regulome (TF-level edges only for
  TFs passing an internal split-half specificity gate; others through modules), TF-TF backbone with
  co-binding and physical views, R2 target score, validation read once after freeze.
- **Gate 1 v1** (`regulatome/gate1/`): 36/79 pass raw × ABC, but all 16 TFs with ≥ 9.1k cells pass
  (depth-limited) and NEUROD4 passes → superseded by v2/v2.1 (2026-10-29). Imputed split-half leaks
  (one model per TF trained on all its cells) → imputed reported as "pass_not_independent".

### 2026-10-26 → 11-02 — 3D view of the chr16p11.2 hub and other Hi-C examples
- **Models** (`results_3d_model/`): PASTIS PM2 on WT Hi-C (10 kb) → Brackley-style TF-binding polymers
  (OpenMM) → hybrid with Hi-C loop restraints → Hi-C-first + TF model (last variant, user decision).
  Cicero edges are 3-10× closer than distance-matched pairs but ~1.0× vs bound, distance/class-matched
  pairs → no pair-specific effect. Hub figures (loop-edge "flower", 18 loop-edges) are illustration only.
- **Direct Hi-C test** (`results_3d_model/hic_hub_test/`): hub promoter bins 1.43× mutual contact vs
  TF-matched chr16 promoters (p = 0.001), but only 1.08× vs TF-rich bins in the same 1.8 Mb window.
- **SE candidate** (ROSE on ENCODE HEK293 H3K27ac, 342 calls, `results_3d_model/se_hub/`):
  chr19 SE_0027 1.76× vs matched (p = 0.004), but a dot test finds a clear dot for only 5/12 partners
  (3 none, 4 untestable) → modest example; re-screen vs keep vs drop awaits the user. 293T H3K27ac
  (GSE246314) only partly reproduces the call.

### 2026-10-21 → 10-28 — Cicero hubs, Hi-C loop support and complex co-binding re-check
- **Hubs:** promoter concentration of multi-TF hubs is a real feature (user + PI); Hi-C validates
  pairs but is not used as a filter. Exon-only signal at highly expressed genes (SERPINH1, DNAJB1,
  BRD4, RPS3) is a suspected cDNA-carryover / hyper-ChIPable artifact → use promoter/distal examples.
  Main example: chr16p11.2 promoter hub. TF × TF Jaccard shows blocks B1-B3; depth-matched top-1,000
  pairs keep only B2/B3 (cell-count bias). Browser tracks stay on raw coverage.
- **Complex co-binding:** per-domain enrichment correction confirms PRC2 (EZH2-MTF2), MAZ-ZNF143 and
  cohesin; NuRD does not hold (its +0.26 came from RBBP4-RBBP7). Slides use Pearson only.
- **Hi-C:** WT HEK293T map GSE216399, Peakachu loops at 0.7. Cicero: step 1 + 25 kb drop only.

### 2026-10-15 → 10-23 — Genome build, benchmarks, CTCF false positives, housekeeping
- **pmat is hg19** (re-verified, `results_293T_analysis/pmat_genome_check/VERDICT.txt`):
  `data/TF1000cells.pmat.mtx.rds` is hg19, everything else hg38; use
  `results_293T_analysis/tf_network_chrombert/inputs/pmat_bins_hg38.bed`. `run_cicero_cobinding.sbatch`
  and `run_cicero_two_stage.sbatch` default to the hg19 pmat. ChromBERT-style encoding dropped.
- **Chromnitron** (Bo Xia preprint; xujie's 43-TF predictions): imputed vs predicted shows no
  diagonal under any normalization (shared ATAC-driven component). AUROC vs bulk: imputed 0.89, raw
  0.64, ATAC alone 0.90 → imputed ≈ ATAC. Imputed vs raw AUPRC for the 7 TFs with bulk ChIP.
- **CTCF promoter false positives** (`ctcf_promoter_fp/`): the reported promoter-only set is empty in
  our data at ≥ 10 reads; best filter = require the CTCF motif (MA0139.2; AUPRC in ATAC bins 0.65 →
  0.73). CTCF is kept but not showcased (shallow; technique may impair CTCF).
- **Motifs:** raw Codebook run for the 11 rescued TFs with union-ATAC background
  (`motif_analysis/motif_raw_codebook/`, job 6083940).
- **Housekeeping:** `xujie_lab/293T_analysis` emptied; still-used outputs moved to
  `XuLab/results_293T_analysis/` and 19 scripts repointed. Lab share `data_share/293T_TF` (537 files,
  md5-verified) replaces the old Globus share. Fragment beds = single-end 5' ends, not Tn5-shifted,
  not deduplicated (~2.6% dups).

### 2026-10-25 — De novo on lab hg38 MACS2 peaks fails its controls: sc peaks carry no own-motif signal
- **Context:** the pmat de novo runs were invalid (Sep: hg19 coordinates on hg38 sequence) or
  accessibility-dominated (Oct 16 hg38-lifted rerun: AP-1/IRF/NRF1 bests, only GC/CpG-like
  candidates). Next route: de novo on the lab's hg38 MACS2 calls
  (`/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/`, p<0.01, no control).
- **Change:** `motif_analysis/denovo_macs2/`. Sites = summits with p<=1e-5, chr1-22/X, minus
  ENCODE blacklist, RefSeq coding exons/3'UTRs (cDNA carryover) and sites covered by strong
  peaks of >50% of the 79 TFs; top 5k per split, summit +/-50 bp. Discovery on odd chr + X
  (HOMER `-cpg` vs the TF's union-ATAC background, STREME vs GC-matched ATAC); candidates ranked
  by AUROC on held-out even chr (GC-matched pos/neg), TOMTOM vs HOCOMOCO v12 + Codebook + JASPAR,
  Codebook artifact flag. 34 still-motif-less TFs + CTCF/NRF1/NFYA/ZNF143/MAZ controls.
- **Outcome:** controls failed: own motif in the top 3 for 0/5 (NFYA rank 6; NRF1 94; MAZ 80;
  CTCF and ZNF143 never recovered). Every TF's best held-out AUROC was 0.55-0.63; top hits were
  Alu fragments (GCCTCAGCCTCC, TGGGATTACA) and SP/KLF GC motifs. Only "novel" candidates:
  4 GC/CpG-like motifs for RBBP4. The shared-site filter removed almost nothing (strong summits
  are covered by 1.7-7 TFs on average), so the accessibility wash is not the cause.
- **Diagnostic (`diag_controls.sh`, `out/diag/diag.tsv`):** own known motif, held-out AUROC vs
  GC-matched open chromatin. Bulk CTCF majority-2-of-3 peaks: **0.72** (evaluation works). sc
  MACS2 sites, whatever the selection (strict / top 2k by p / top 2k by fold / +/-250 bp):
  CTCF 0.39-0.47, NRF1 0.49-0.50, ZNF143 0.51-0.52, MAZ 0.48-0.50, NFYA 0.55-0.64. sc sites are
  13-25% Alu vs 2% for bulk CTCF.
- **Audit, all 49 TFs with a database motif (2026-10-26, `audit_own_motifs.sh`, `out/audit/`):**
  best own-motif AUROC on the strict sc sites is 0.44-0.55 vs open chromatin and 0.48-0.56 vs
  GC-matched random genome; only NFYA reaches 0.55-0.56. Bulk CTCF scores 0.72 / 0.78 the same way.
  The sc MACS2 sites carry no own-motif signal for any panel TF, not just the controls.
- **Artifact drop (2026-10-26, `audit_artifact_drop.sh`, `out/audit/artdrop/`):** removing sites that
  overlap an Alu or carry a FIMO hit (p<1e-5) to the 18 non-endogenous Codebook artifact motifs
  (NFI/SP-KLF/ETS/NR kept), from both foreground and backgrounds, changes nothing: own-motif AUROC
  0.49-0.55 vs genome, 0.43-0.55 vs open chromatin (NFYA best); bulk CTCF 0.77/0.72. Artifact
  motifs hit almost no sites (71 of 1.45M); Alu overlap removed 6-14%.
- **Dedup re-call (2026-10-28, `recall_dedup.sbatch`, `out_dedup/`):** 5 controls re-called with macs3, lab
  settings, reads deduplicated per cell (chrom + 5' end + strand + barcode). Dedup removes only 1-3% of reads
  (25% inside the old CTCF strong sites); CTCF strong summits 4.3k -> 2.5k. Own-motif AUROC unchanged:
  vs open chromatin CTCF 0.48, NRF1 0.49, ZNF143 0.51, MAZ 0.52, NFYA 0.55; vs genome 0.50-0.56;
  bulk CTCF 0.73/0.78. Duplicates are not the cause; not extended to the panel.
- **Decision:** the lab sc MACS2 peaks do not localise to TF motifs for most TFs, so no de novo
  candidate from them is taken forward. Next: score every known-motif TF's own motif the same
  way to find which TFs carry motif signal, and revisit the motif enrichment setup (Codebook
  artifact handling, controlled peak calls) before more de novo work.
- **Files:** `motif_analysis/denovo_macs2/{prep_sites.py,prep_sites.sbatch,run_denovo.sbatch,summarize.py,summarize.sbatch,submit_all.sh,diag_controls.sh,tfs.txt}`;
  results `motif_analysis/denovo_macs2/out/{summary.tsv,novel_candidates.tsv,sites/site_counts.tsv,diag/diag.tsv}`.

### 2026-10-12 — Housekeeping: TSV results out of git
- `.gitignore` now ignores `*.tsv` / `*.tsv.gz` (except `metadata/*.tsv` and `*_manifest.tsv`).
  TSVs already tracked stay tracked until removed with `git rm --cached`.

### 2026-10-09 — Cicero "Our Run" rolled out to the panel; 2 Mb window confirms RBBP4 loop enrichment
- **Panel run:** `slurm/run_ourrun_tf.sbatch` (array over `configs/ourrun_tfs.txt`, 78 TFs,
  RBBP4 excluded since its run exists) applies the RBBP4 "Our Run" settings (entry below) to
  every TF. Peak matrix is built per TF from the lab's MACS2 narrowPeak (q<=0.05) + fragment
  BED (`scripts/build_tf_peak_mm_from_macs2.py`; `build_tf_peak_mm.py` = generic variant), then
  Cicero 1 Mb → p<=0.05 on the RBBP4 Gaussian null (`RBBP4.perm.fitConns.para.txt`) → FDR<=0.05
  cliques → loop metrics + permutation. Outputs `work/ourrun_k50/<TF>/`, `results/ourrun_k50/<TF>/`.
- **Status (array 4433127, as of 2026-10-12):** 67/78 tasks finished; selected pairs per TF
  median ~1.1k, max ~135k; 7 tasks logged errors → check `logs/ourrun-4433127_*.err` and
  resubmit those indices. Using RBBP4's null for every TF is a shortcut; a per-TF null is the
  stricter follow-up.
- **RBBP4 2 Mb second search:** `run_rbbp4_window2mb.sbatch` reruns Cicero at a 2 Mb window
  (same k=50 / cap 10k / seed 555) and keeps pairs selected in BOTH windows (12,433 edges,
  359 cliques at FDR<=0.05). `run_rbbp4_w2mb_metrics.sbatch` scores them against Peakachu loops.
- **Loop-overlap slide table** (`hic/scripts/fill_slide_loop_table.py`, `run_rbbp4_slide_perm.sbatch`):
  locked loop definition ±30 kb, score>=0.5 (5 kb + 10 kb loops), optional drop of spans <=25 kb,
  500-draw same-chromosome permutation (cliques shifted as one block). Both-windows RBBP4 result:
  pair loop rate **16.0% vs null 9.4%**, cliques with >=1 loop-supported edge **43.5% vs 25.1%**,
  all edges supported **3.1% vs 0.8%** (all p=0.002, 0/500 null draws >= observed). With spans
  <=25 kb dropped: 16.1% of pairs, 40.8% of cliques with >=1 supported edge. The pairs are
  loop-enriched ~1.7× over a position-shuffled null even though most pairs are not loops.
- **Files:** `cobinding/slurm/{run_ourrun_tf,run_rbbp4_window2mb,run_rbbp4_w2mb_metrics,run_rbbp4_slide_perm}.sbatch`,
  `cobinding/scripts/build_tf_peak_mm{,_from_macs2}.py`, `cobinding/configs/ourrun_tfs.txt`,
  `hic/scripts/{fill_slide_loop_table,plot_cicero_loop_cliques,plot_pairs_vs_loops_chromosomes}.py`.

### 2026-10-07 — RBBP4 Cicero parameter sweep: k=50 / cap 10k / seed 555 chosen as "Our Run"
- **Context:** on the exact RBBP4 peak matrix/CDS, sweep the aggregation knobs (k, metacell
  iteration cap, seed; 1 Mb window, p<=0.05 shuffle null) and compare each to Dr. Wang's gold
  (17,450 pairs; 11,342 FDR<=0.05 edges; 355 cliques). `slurm/run_rbbp4_parameter_sweep.sbatch`.
- **Result (selected pairs / cliques; Spearman of coaccess on shared pairs):**

  | k | cap | seed | pairs | cliques | gold-FDR recovered | Spearman |
  |---|---|---|---|---|---|---|
  | 40 | 10k | 555 | 17,839 | 277 | 12.7% | 0.33 |
  | 50 | 8k | 555 | 19,839 | 379 | 13.6% | 0.33 |
  | **50** | **10k** | **555** | **20,613** | **359** | **14.3%** | **0.31** |
  | 50 | 12k | 555 | 5,332 | 66 | 5.6% | 0.29 |
  | 50 | 15k | 555 | 19,694 | 325 | 14.4% | 0.29 |
  | 50 | 10k | 2020 | 18,563 | 308 | 13.4% | 0.33 |
  | 60 | 10k | 555 | 22,671 | 497 | 15.1% | 0.30 |
  | 75 | 10k | 555 | 24,835 | 654 | 15.9% | 0.28 |

- **Outcome:** k=50 / cap 10k / seed 555 matches the gold network's SIZE best (359 vs 355
  cliques, largest module 6 vs 5) and became the "Our Run". Pair-level agreement stays low at
  every setting (Jaccard ~0.06, Spearman ~0.3): the counts can be matched, the exact pairs
  cannot, consistent with metacell sampling being intrinsically unstable. Higher k inflates
  cliques. The cap=12k point is an unexplained outlier (5k pairs); rerun before trusting it.
- **Files:** `cobinding/slurm/run_rbbp4_parameter_sweep.sbatch`; results in
  `cobinding/results/RBBP4_sweep_k*_i*_s*/compare_to_gold.txt`.

### 2026-10-07 — Codebook motifs rescue 11 previously motif-less TFs
- **Change:** AME (`--scoring avg`) of each TF's peaks/bins against `codebook_top1.meme`
  (`motif_analysis/slurm/motif_enrichment_codebook.sbatch`, array 0-10; genome/OCR backgrounds,
  peaks and bins). `scripts/fill_codebook_known_sheet.py` writes the own-motif adj. p into the
  Codebook-only rows of `TF1000cells_motifs.xlsx`.
- **Outcome:** 11 TFs moved from "unknown" to "known motif": ZNF507, MBD3, ZNF703, GLYR1,
  FLYWCH1, ZNF367, ZNF746, POGK, ZBTB40, ZNF606, ZNF503. Shrinks the set for the novel-motif
  module (Steps 4-5).
- **Commits:** `2d1d6a9`, `3ddd900`.

### 2026-10-05 → 10-09 — 10 kb Raw / Imputed / Chromnitron / Bulk Pearson per TF
- **Context:** per-TF genome-wide check of how the imputed aggregate relates to the raw
  aggregate, the Chromnitron sequence prediction (`HEK293T_SA`) and bulk ChIP.
- **Change:** `downstream/tf_hitag_pearson/build_aggregate_matrix.py` (10 kb bins, SLURM array
  `tf_aggregate_pearson.sbatch`) → `build_bin10kb_matrices.py` assembles per-TF 4×4 matrices
  plus raw-vs-imputed (all TFs) and imputed-vs-pred (TFs with a Chromnitron track; 36 lack one,
  `bin10kb/tfs_without_pred.txt`). `apply_spqn_imputed_pred.py` applies the compartment-style
  SpQN (cell-count covariate) to imputed-vs-pred, diagonal kept as the matched Pearson.
  1 kb variant in `build_bin1k_correlations.py` (plot layout enlarged). Old per-TF result
  files under `ctcf_hitag_pearson/` and `tf_hitag_pearson/` were removed from git (`8743036`).
- **Example (CTCF):** Raw–Imputed r=0.71; Pred–Bulk 0.67; Raw/Imputed vs Bulk ~0.37 and vs
  Pred ~0.30. Imputation tracks its own raw input far more than the external references.
- **Files:** `downstream/tf_hitag_pearson/*.py`, `downstream/slurm/tf_{aggregate_pearson,pearson_bin1k}.sbatch`.

### 2026-10-05 — Cicero co-binding: two-stage reproducibility filter (vs Hi-C loops)
- **Motivation:** single-round Cicero coordinated-binding pairs barely overlap the Hi-C
  Peakachu loops (`hic/scripts/compare_loops_to_cicero_pairs.py`), suggesting many pairs may
  be spurious. Test reality by **reproducibility**: re-run Cicero RESTRICTED to the peaks in
  round-1's called pairs, call pairs again, keep only pairs significant in BOTH rounds.
- **Built (opt-in, NON-destructive — round-1 outputs untouched):**
  - `scripts/extract_pairs_peaks.py` — unique peaks from a q-filtered peak_edges.tsv.
  - `export_pmat_tf_mm.R` — new `--regions` to subset the pmat to those peaks (additive; no
    effect on existing calls).
  - `scripts/intersect_pair_edges.py` — round1 ∩ round2 unordered pairs → reproducible set
    (+ round-2 coaccess/qval), prints Jaccard / % of round1.
  - `slurm/run_cicero_two_stage.sbatch` — orchestrates: extract pair-peaks → restricted mm →
    Cicero round 2 (+ genbsp null, same shuffle-Gaussian significance) → intersect → cluster.
    Writes only `work/<TF>/cicero_stage2/` + `results/<TF>/stage2/`. Optional `LOOP_CORE/MID/
    LOOSE` runs the loop-overlap compare on round1 vs reproducible to show if it improves.
- **Run:** round 1 first (`run_cicero_cobinding.sbatch`), then
  `TF=RBBP4 sbatch cobinding/slurm/run_cicero_two_stage.sbatch`.
- **Files:** `cobinding/scripts/{extract_pairs_peaks.py,intersect_pair_edges.py,export_pmat_tf_mm.R}`,
  `cobinding/slurm/run_cicero_two_stage.sbatch`.


### 2026-10-02 — Got the Codebook data (Zenodo 15667805); wired its motifs + artifact set
- Paper: Hughes/Codebook Consortium, "An expanded codebook of human TF DNA-binding
  specificity" (Nature 2026, s41586-026-10798-9); data on Zenodo 10.5281/zenodo.15667805,
  browser mex.autosome.org (same autosome.org group as HOCOMOCO v12).
- Built `scripts/fetch_codebook.sh` (Zenodo fetcher) + `scripts/ppm_to_meme.py` (.ppm/.pcm ->
  MEME). Produces: `codebook_artifacts.meme` (37 empirical artifact motifs: poly-G, Alu/repeat,
  CAC/GGAA, NFI, self-annealing -- the "use their set" from the slide, ready MEME) and
  `codebook_top1.meme` (204 curated motifs, 1 per TF) for the v12+Codebook re-annotation DB.
- Step 1 now complete (DB build; MoSBAT still the optional upgrade over tomtom-pearson).
- Step 2 upgraded: `flag_motif_artifacts.py --artifact-tomtom` flags de-novo motifs matching
  the Codebook artifact set (caught a GGAGGG self-annealing motif that name-regex missed);
  `rank_motifs.sbatch` runs TOMTOM vs both the merged DB and the artifact DB.
- CHS=1 pulls MEX.CHS.tar (1.4 GB ChIP-seq peaks) for Step 4; GHT-SELEX on zenodo 8327970.
- Files: `motif_analysis/scripts/{fetch_codebook.sh,ppm_to_meme.py,flag_motif_artifacts.py}`,
  `motif_analysis/slurm/rank_motifs.sbatch`, `motif_analysis/NOVEL_MOTIF_FINDING.md`.

### 2026-10-02 — Novel Motif Finding Step 2: contaminant artifact filter built
- `scripts/flag_motif_artifacts.py` flags de-novo motifs whose best TOMTOM match (vs
  HOCOMOCO v12 + Codebook) is an open-chromatin contaminant family (CTCF/NFY/YY1/SP-KLF/ETS;
  `--families` takes the Codebook set verbatim). Joins onto the Step-3 ranked table and adds
  a `pass` column -> shortlist = high AUROC AND pass. Optional TE/repeat check via
  `--repeat-bed`+`--instances-bed` (needs RepeatMasker, not yet available). Wired into
  `rank_motifs.sbatch` (TOMTOM candidates vs merged DB -> flag -> ranked_annotated.tsv).
  Validated on synthetic: CTCF/SP1 matches flagged out despite high AUROC; novel Codebook
  ZNF passes. User has HOCOMOCO v12 + Codebook available (for merge_meme_db.py).

### 2026-10-02 — Novel Motif Finding for unknown TFs: start module (Step 3 = perf ranking built)
- **Goal:** turn raw de-novo "unknown" motifs (from `motif_denovo_pmat.sbatch`: HOMER +
  STREME + TOMTOM for ~45 motif-less TFs) into validated, TF-assigned motifs via a 5-step
  plan. Roadmap in `motif_analysis/NOVEL_MOTIF_FINDING.md`.
- **Built — Step 3 (the centerpiece): rank motifs by PERFORMANCE, not information content.**
  `scripts/rank_motifs_by_performance.py` scores every pos/neg sequence by a motif's best
  PWM log-odds (both strands) → **AUROC + AUPRC**, and reports info content so a low-IC
  "flat" motif with high AUROC is visible. Validated on synthetic (real AUROC 1.0, flat 0.5).
  `slurm/rank_motifs.sbatch` runs it per TF: scoring set = the TF's binding BINS (independent
  of the pmat PEAKS used for discovery → held-out-ish), candidates = HOMER de-novo + STREME.
- **Built — Step 1 helper:** `scripts/merge_meme_db.py` merges MEME DBs (tag by source) to
  rebuild `cache/motifdb/merged_human_motifs.meme` as **HOCOMOCO v12 + Codebook** (replacing
  v11) for TOMTOM re-annotation. Still need: v12/Codebook download URLs; MoSBAT (affinity)
  as the upgrade over TOMTOM pearson.
- **Not yet (need external data):** Step 2 contaminant filter (CTCF/NFY/YY1/SP-KLF/ETS +
  RepeatMasker), Step 4 triple-overlap (Codebook ChIP/GHT-SELEX peaks), Step 5 orphan→TF
  assignment (chromVAR/MARA + TF expression + phyloP, 175 motif-less TFs).
- **Files:** `motif_analysis/scripts/{rank_motifs_by_performance.py,merge_meme_db.py}`,
  `motif_analysis/slurm/rank_motifs.sbatch`, `motif_analysis/NOVEL_MOTIF_FINDING.md`.

### 2026-10-02 — Found the actual Ren-lab Cicero pipeline; aligned our params (explains the score gap)
- **Source:** Zu 2023 Nature & Li 2021 Nature (Songpeng Zu / Yang Eric Li; **Kangli Wang**
  co-author = our colleague). Code: `github.com/yal054/snATACutils`,
  `bin/snapATAC.runCicero.R` (real) + `bin/snapATAC.runCicero.shuf.R` (nulls).
- **Why our coaccess didn't match his (Spearman 0.13–0.30) — parameter mismatch, not just
  intrinsic noise:**
  | param | Ren-lab | ours (was) |
  |---|---|---|
  | cells | **downsample to 1000** (seed 2020) | ALL (RBBP4 163k) |
  | `make_cicero_cds(k=)` | **10** | 50 |
  | `run_cicero` window | **500 kb** (default) | 1 Mb |
  | seed | **2020** | 555 |
  | p-value null | **`perm`=genBsp random bootstrap** (+row/col/both shuf) → Gaussian(μ,σ) → upper-tail p → BH | colshuf only |
  Confirmed by the colleague's data path `.../Cell1000/cicero/` and `RBBP4.perm.fitConns.*`.
  Corrects my earlier `n≈8500` (that assumed a correlation t-test; the real significance is
  the **shuffle Gaussian** on the genBsp null, so `n` never entered).
- **Change — align to Ren-lab:** new `cobinding/configs/cicero_renlab.yaml`
  (max_cells 1000, k_metacell 10, window_bp 500000, seed 2020); `03_run_cicero.R` gains the
  `genbsp` null (exact `genBsp`: random bootstrap, matched nnz, double `set.seed`);
  `run_cicero_cobinding.sbatch` now defaults `CFG=cicero_renlab.yaml` + `SHUFFLE_MODE=genbsp`.
- **Next:** rerun RBBP4 `IGNORE_FITCONNS=1 REBUILD_CICERO=1` on the renlab config, then
  `compare_raw_cicero_conns.R` vs his `RBBP4.conns.rds` — Spearman should rise well above
  0.13–0.30 now that cells/k/window/null match. Exact bit-match still limited by which 1000
  cells + UMAP seed, but the significance FRAMEWORK and param set now mirror his.
- **Also available in snATACutils (not yet used):** `run.predictGenePeakCorrJoint.R`
  (gene–peak correlation), `generate_ccans` (CCANs) — relevant if we extend beyond pairs.
- **Files:** `cobinding/configs/cicero_renlab.yaml`, `cobinding/cicero/scripts/03_run_cicero.R`,
  `cobinding/slurm/run_cicero_cobinding.sbatch`.

### 2026-10-02 — Cicero replication: wire the Ren-lab shuffle null into the main pipeline (exact match only from his table)
- **Question:** can we replicate the colleague's RBBP4 pairs AND p-values exactly?
- **Answer:** **Exact replication only from HIS table** (`RBBP4.perm.fitConns.res.txt`
  coaccess + `*.para.txt` meanShuf/stdShuf), not by rerunning Cicero. Already wired:
  `run_rbbp4_labconns.sbatch` + `cicero_conns_to_edges.py --pval-method {precomputed,shuffle}`
  reproduce his 17,450 `.sel` / 11,342 FDR≤0.05 / 355 cliques. Cicero itself emits no
  p-values; his **permutation (shuffle) null** does, and we matched it.
- **Why not from our own run_cicero:** same pmat gives ~96% of his candidate pairs but the
  **coaccess ranking disagrees (Spearman ~0.13–0.30)**. Cicero coaccess is set by the
  **metacell aggregation** (LSI→UMAP embedding + kNN sampling in `make_cicero_cds`), which is
  not reproducible without his exact embedding/seed — likely intrinsic (two seeds of our own
  run would also disagree). So a p-value filter on *our* Cicero will not return his `.sel`.
  Diagnostic to confirm intrinsic-ness: run our Cicero twice with different seeds, Spearman
  the coaccess (`compare_raw_cicero_conns.R`).
- **What we CAN do for all TFs (not just RBBP4):** apply his significance **methodology** to
  our own Cicero. The infra existed (`03_run_cicero.R` shuffle modes colshuf/rowshuf/perm →
  Gaussian `shuffle.para.txt`; patched `make_cicero_cds` cap → ~8.5k metacells) but the main
  per-TF pipeline defaulted to the t-test. **Change:** `03_run_cicero.R` now takes `--shuffle`
  (non-'none' writes to `cicero_shuf/`); `run_cicero_cobinding.sbatch` runs the real + shuffle
  Cicero and sets `SIG_METHOD=shuffle` (colshuf null → Gaussian p → BH, filter FDR≤0.05) by
  default, so every TF gets Ren-lab-style significance. `SIG_METHOD=ttest` keeps the old path.
- **Bottom line:** his EXACT RBBP4 numbers → from his table; his METHOD (shuffle null + BH) →
  now applied to our Cicero for all TFs. Bit-matching his per-pair scores is out of reach.
- **Files:** `cobinding/cicero/scripts/03_run_cicero.R`,
  `cobinding/slurm/run_cicero_cobinding.sbatch` (+ existing `run_rbbp4_labconns.sbatch`,
  `cicero_conns_to_edges.py`, `compare_raw_cicero_conns.R`).

### 2026-09-24 — Enrichment-ratio RESULT: real complexes separated; several earlier "hits" were activity artifacts
- **Ran** `CORR_TRANSFORM=enrichment` (domain units, results_compartment_enrichment_domain_sm2).
  Baseline collapsed: off-diag **median +0.85 → −0.02** (min −0.66, p95 +0.32); confound
  `corr(log10 cells, mean_corr)` = +0.13. Heatmap = ~0 (blue) background with discrete red
  co-occupancy blocks — a real discriminator vs the old all-red activity matrix.
- **CONFIRMED complexes (score = mean pairwise enrichment-r above ~0 background):**
  - PRC2 `EZH2/MTF2` = **+0.67** (cleanest, strongest) ⭐
  - boundary `MAZ/ZNF143` = **+0.35** (and `+CTCF` = −0.10 → CTCF-independent) ⭐
  - cohesin `CTCF/STAG2` = **+0.28**
  - NuRD `RBBP4/RBBP7/MBD3` = **+0.26** — **rescued** (the activity baseline had masked it)
- **CORRECTION — earlier percentile-based "hits" were shared-territory/activity artifacts,
  they COLLAPSE under enrichment:** `WIZ/ZNF644` (G9a) +0.03, `FOXK1/MLX` +0.04,
  `DR1/POGK/ZNF746` +0.07, repressive super-module +0.07. **Do NOT motif-confirm these as
  complexes** — supersedes the 2026-09-23 shortlist.
- **Dominant emergent module (caveat):** a large `RBBP4/XPA/E2F3/DDIT3/RBBP7/CTCF/SAFB/ZNF703`
  clique (mean +0.65; RBBP4-XPA +0.91, E2F3-RBBP4 +0.88) — an RbAp48/DREAM/architecture hub.
  BUT these are the **highest-cell TFs**, and the confound is +0.13 (nonzero), so this may
  carry **residual complexity bias**. Rule it out before claiming: rerun
  `CORR_TRANSFORM=enrichment SPQN=1` (enrichment kills the activity baseline, SpQN the
  residual cell-count effect) and/or check at peak scale.
- **New candidate modules to look at:** `HOXB9/NME2` (+0.69), `MBD3/NME2/HOXB9/DR1` (+0.54),
  `CHAMP1/ZBTB43/ZBTB7A` (+0.42).
- **NEXT (cluster agent):** motif-confirm the CONFIRMED list (PRC2, MAZ/ZNF143, cohesin,
  NuRD) — not the collapsed ones; then re-score after `enrichment + SpQN` to settle the
  high-cell RBBP4/XPA/E2F3 hub. Data: `tf_similarity_genome.tsv` in the results dir.
- **Files/inputs:** `results_compartment_enrichment_domain_sm2/tf_similarity_{genome,A,B}.tsv`.

### 2026-09-24 — Compartment matrix is activity-dominated, not cell-count-confounded; add enrichment-ratio transform
- **Context:** The raw compartment TF×TF matrix (RPKM, colored by raw Pearson r,
  vmin 0.80) shows a solid all-red top block that *looked* like the complexity confound.
- **Finding — it is NOT the cell-count confound.** `corr(cells, mean_corr) ≈ 0` and the
  red block is **depth-mixed** (RBBP4 181k cells sits with SETDB1/ZNF503/ZNF606 ~1k, all
  ~0.97). The slab is the **A/B compartment-activity baseline**: every TF puts more reads
  in active A domains → all TFs correlate ~0.9 via that shared shape. Compartment-scale
  aggregation makes even 1k-cell TFs well-measured, so depth doesn't drive the correlations
  here. The "cleaner" percentile-colored plots differ from the raw-r plot **only by coloring**
  (rank-spread hides the high baseline), not by real de-confounding.
- **Correction to the entry below:** the "SpQN removed the confound (→0)" reading measured
  the file `compartment_counts_genome.tsv`, which is the **RAW** matrix (already `corr≈0`).
  SpQN did **not** do the de-confounding at compartment scale — it's ~a no-op there because
  there was little cell-count bias to begin with. SpQN's real value is at **peak scale**
  (the original `tf_complexes` all-red rich-club), where features are sparse/noisy.
  RPKM/per_cell (per-TF) and SpQN (cell-count) all leave the activity baseline untouched.
- **Change:** added a per-domain **enrichment ratio** (observed/expected): each domain ÷ its
  across-TF mean, subtracting the shared activity baseline so each TF's *deviation* (where
  complexes live) drives Pearson. It is a per-DOMAIN op → it actually changes the correlation
  (unlike the per-TF no-ops). `compartment_rpkm_correlation.py --transform enrichment`;
  sbatch `CORR_TRANSFORM=enrichment` (build NORM stays rpkm).
- **Validation (synthetic: activity baseline + planted complex):** raw median **+0.85**,
  complex−background gap **+0.02** → enrichment median **0.00**, gap **+0.50**. Baseline
  removed, complex separates cleanly.
- **Complexes still valid:** PRC2, cohesin, WIZ+ZNF644, MAZ+ZNF143, etc. are real
  co-occupancy (visible in raw too) — just not "SpQN-revealed."
- **Next:** run `CORR_TRANSFORM=enrichment`; check whether modules tighten / new ones emerge
  above a now-zero baseline; keep SpQN for the peak-level analysis.
- **Files:** `tf_complex/scripts/compartment_rpkm_correlation.py`,
  `tf_complex/slurm/05_compartment_rpkm.sbatch`.

### 2026-09-23 — SpQN result: complexity confound removed + new candidate complexes (for motif confirmation)
> **CORRECTED 2026-09-24 (see entry above):** the matrix measured here was the RAW compartment
> matrix (already `corr(cells,mean_corr)≈0`). SpQN did **not** do the de-confounding — the
> compartment matrix is *activity*-dominated, not cell-count-confounded. The candidate
> complexes below remain valid (real co-occupancy); ignore the causal claim that SpQN
> revealed them.
- **Status:** SpQN on the A/B-compartment TF×TF correlation **removed the cell-count
  confound.** On the delivered genome-scope matrix, `corr(cells, mean_corr) = +0.00`
  (linear) / `+0.09` (vs log10 cells) ≈ 0. Concrete proof: RBBP4 (181,396 cells)
  mean-corr **0.44** vs HOXC4 (1,023 cells) **0.58** — the deepest TF is *less* sticky
  than a shallow one, so the "deep-TF rich club" is gone. (Still want the pre-SpQN
  `+X` from the `05_compartment_rpkm` `.out` `confound` line to headline the delta.)
- **Data analyzed:** `compartment_counts_genome.tsv` = the SpQN-corrected TF×TF matrix,
  **percentile-transformed** (off-diag quantiles uniform `[.05,.25,.50,.75,.95]`), 79 TFs.
  Values below are mean pairwise **percentile** of Pearson r (background median = 0.50;
  ≥0.90 = strong co-occupancy). Method: Bron–Kerbosch cliques + reciprocal-best-partner
  pairs + per-module mean-percentile scoring.

- **Validation (positive controls):**
  - PRC2 `EZH2+MTF2` = **0.98** ✓
  - Cohesin/insulator `CTCF+STAG2` = **0.93** ✓
  - NuRD core `RBBP4+RBBP7+MBD3` = **0.60** ✗ — NuRD does NOT co-occupy at compartment
    scale (expected; NuRD is peak-scale → check via the Cicero/cobinding track).

- **NEW candidate complexes/modules (ranked; to confirm by motif co-enrichment):**
  1. **G9a/GLP H3K9-methylation** — `WIZ + ZNF644` = **0.98** (mutual best partners).
     Both are documented G9a–GLP (EHMT2/EHMT1) components → strongest, literature-backed
     de-novo hit.
  2. **Repressive-chromatin super-module** — `WIZ + ZNF644 + MTF2 + SMARCA1` = **0.99**
     (H3K9 + H3K27/PRC2 + ISWI remodeling co-occupying the same territory).
  3. **CTCF-independent boundary** — `MAZ + ZNF143` = **0.98**, while `MAZ+ZNF143+CTCF`
     = **0.40** (CTCF does NOT join → matches MAZ-as-CTCF-independent-boundary reports).
  4. **Metabolic** — `FOXK1 + MLX` = **0.99** (extends to SREBF2 = 0.89). MLX↔MondoA/ChREBP,
     FOXK1 glycolysis regulator.
  5. **Basal/NC2 repressor** — `DR1 + POGK + ZNF746` = **0.98** (DR1 = NC2β).
  6. **NuRD/DREAM-adjacent pairs** — `RBBP7 + ZNF703` = 0.98 (ZNF703 recruits NuRD);
     `E2F3 + RBBP4` = 0.998 (RbAp48 in Rb–E2F/DREAM).
  - Other strong reciprocal pairs to sanity-check: `SRCAP–MLX` 0.998, `NFYA–AKAP8L` 0.984,
    `MBD3–ZNF777` 0.962, `NME2–NFYA` 0.959, `SOX4–TRAFD1` 0.996.

- **Caveats (state in any writeup):** compartment-scale co-occupancy = shared large-scale
  territory, **not** proof of physical interaction; some high pairs are shared-territory
  coincidence (e.g. `RBBP4–XPA` 1.00). The **A/B-activity confound** still shapes the big
  blocks — trust the *within-block tight pairs above*, not the blocks. SpQN can over-flatten
  at n=78; validation complexes surviving is the reassurance.

- **NEXT (cluster agent):** confirm the ranked modules by **motif co-enrichment** — do the
  paired TFs' motifs co-occur in the same regions above background? Prioritize the two clean
  literature-backed hits **WIZ+ZNF644 (G9a/GLP)** and **MAZ+ZNF143 (CTCF-independent
  boundary)**. Cross-check all strong pairs against CORUM/BioGRID (annotated vs novel), and
  against the peak-level Cicero/cobinding results (esp. NuRD, which should appear there but
  not here). Motif pipeline: `motif_analysis/slurm/motif_enrichment.sbatch` (AME `--scoring avg`).
- **Files/inputs:** `compartment_counts_genome.tsv`, `data/TF1000cells.meta.csv`,
  `tf_complex/scripts/compartment_rpkm_correlation.py` (`--spqn`).

### 2026-09-23 — Compartment correlation: cell-count normalization is a no-op; wired SpQN (with caveat)
- **Context:** The A/B-compartment TF×TF correlation has a complexity bias — TFs
  with more cells correlate with each other. PI suggested normalizing by cell count.
- **Finding:** Dividing each TF by its cell count is a **per-TF scalar → Pearson
  scale-invariant → no-op** (the new `per_cell` transform gives a byte-identical
  matrix to RPKM; same as CPM). The bias is not magnitude but **differential
  measurement noise**: low-cell TFs have noisy profiles → attenuated correlations;
  high-cell TFs are clean → correlate strongly with each other. Scaling can't touch
  signal-to-noise. Real fixes: equalize cells by **downsampling**, or **SpQN**.
- **Change:** Wired **SpQN** (Wang/Hicks/Hansen 2022) into
  `compartment_rpkm_correlation.py` (`--spqn`, reuses the vetted `tf_complexes.spqn`,
  covariate = per-TF cell count) + `_confound` diagnostic (`corr(cells, mean_corr)`
  before/after) + `SPQN=1` sbatch knob. Writes `spqn_similarity_<scope>.tsv`;
  clusters/plots the corrected matrix.
- **Caveat (validated on synthetic):** at ~78 TFs SpQN is coarse and its
  level-to-reference mechanism raised the off-diag median (+0.65→+0.85) while only
  partly cutting the confound (+0.66→+0.48) and **flattening a planted complex**
  (gap 0.18→0.04). So it may make the matrix uniformly red and wash out NuRD/PRC2
  rather than surface them. **Judge on real data via the diagnostic + whether
  NuRD (rbbp4/rbbp7/mbd3) and PRC2 (ezh2/mtf2) stand above background;** if not,
  fall back to downsample-to-equal-cells. Separate confound (all TFs prefer active
  compartments) still needs a per-domain enrichment ratio.
- **Files:** `tf_complex/scripts/compartment_rpkm_correlation.py`,
  `tf_complex/slurm/05_compartment_rpkm.sbatch`.

### 2026-09-23 — Cicero co-binding: run on the fragment-called peak matrix (the real fix)
- **Context:** After adding significance (below), a validation run still barely
  matched the lab gold — pair Jaccard 0.028, ~7k regions/5k edges vs gold ~20k/17k.
  Significance was **orthogonal**; the real problem was the **input peak universe**.
- **Diagnosis chain:**
  - Our cobinding Cicero was fed peaks from `call_raw_peaks.py`, which calls MACS on
    the **1 kb-bin** pseudobulk (synthetic fragments at bin midpoints). That quantizes
    peaks to a 1 kb grid → **0 exact coordinate matches** with the lab's fragment
    peaks, and ~⅓ as many sites.
  - The lab's gold peaks are variable-width fragment MACS peaks (min 150, median 262).
    Resolution itself is *not* the blocker: gold peaks collapse onto 1 kb bins nearly
    losslessly (19,199 bins, 0.6% pairs lost) — so a bin-level analysis *could*
    recover ~99%; the gap was sensitivity/wrong sites, not the 1 kb grid.
  - We do **not** have raw fragments. But the colleague sent a **fragment-called peak
    matrix**, `data/TF1000cells.pmat.mtx.rds` (237,450 peaks × 885,203 cells, all TFs
    pooled, mean width 412 bp) — the correct universe we'd been ignoring.
- **Change:** New `cobinding/scripts/export_pmat_tf_mm.R` subsets the pmat to a TF's
  cells (barcodes from `unified/work/<tf>/mm/barcodes.tsv.gz`), filters
  `min-cells-per-peak`, and writes the `mm/{matrix,regions,barcodes}` trio that
  `02_build_cds.R` consumes. `run_cicero_cobinding.sbatch` now calls it instead of
  `export_tf_cicero_mm.py` (1 kb-bin path); added `PMAT`, `MIN_CELLS_PER_PEAK`,
  `MIN_PEAKS_PER_CELL`, and `IGNORE_FITCONNS=1` (force the pmat path on RBBP4 to
  compare vs gold). Validated the exporter end-to-end locally (valid mm, 100% barcode
  match, filters applied).
- **Open items / caveats:**
  - **pmat vs gold peak universe barely overlaps** (0 exact, 12% ≥1 bp, 7.8%
    center-in) — likely a **genome-build mismatch** (fitConns hg19 vs pmat hg38?) or
    the fitConns used a *separate* RBBP4-specific peak set. So a pmat run won't
    bit-for-bit reproduce fitConns; validate by network quality/significance, or get
    the colleague's exact inputs. **Next: liftOver test to settle the build question.**
  - Barcode-format match between `unified` mm and the pmat is asserted at runtime
    (fails loud) but unverified until a cluster run.
  - Cell-count parity unresolved: RBBP4 = 181,396 pmat cells vs a *back-calculated*
    (assumption-laden, possibly wrong) lab n≈8.5k. Not treating 8.5k as fact.
  - Non-fitConns path still keeps all edges at cluster time (`CLUSTER_QVAL=1`); wiring
    the new FDR into that filter is a deliberate follow-up.
- **Files:** `cobinding/scripts/export_pmat_tf_mm.R` (new),
  `cobinding/slurm/run_cicero_cobinding.sbatch`.

### 2026-09-23 — Cicero co-binding: add the missing significance (fitConns pval/qval)
- **Context:** Our co-binding cliques looked far less significant than the lab
  colleague's `fitConns` workbook. Root cause found in our own code: Cicero's
  `run_cicero()` emits **only** a `coaccess` score (regularized partial
  correlation, [-1,1]) — **no p-value / FDR**. `cicero_conns_to_edges.py`
  hardcoded `qval=0`, so downstream `q<=0.05` clique filtering was meaningless on
  any TF we ran ourselves. We only matched the reference on RBBP4 because we
  *inherited* the colleague's q-values from their fitConns file. The genuine
  Ren-lab Cicero papers (mouse cerebrum *Nature* 2021; human brain *Science* 2023)
  also just threshold the coaccess score — the per-pair significance is an add-on.
- **Diagnosis:** Reverse-engineered the fitConns `pval` from the reference RBBP4
  data — it's a **correlation-significance test on the coaccess score** (t-test /
  Fisher-z) with effective n ≈ number of cells/metacells (calibrated n≈8,531 for
  RBBP4, ~10k cells), then BH-adjusted. It's a smooth function of `coaccess`
  alone → rules out any permutation or distance-decay null.
- **Change:** `03_run_cicero.R` now records `n_metacell` in `cicero_info.tsv`;
  `cicero_conns_to_edges.py` computes a two-sided t-test (`df=n_eff-2`) on each
  pair's coaccess + BH `qval` (new `--effective-n`, default `metacell`, read from
  the sibling info file). `cluster_peaks.py` now filters on real FDR. Kept it in
  the co-binding path only — the Cicero **imputation** path is retired, untouched.
- **Outcome / caveat:** Model is exact at the q≈0.05 cutoff and the 2.22e-16 floor,
  within ~1 order of magnitude in the mid-range (reference likely uses a per-pair
  n we don't get from `run_cicero`'s output). Good enough for a first pass; refine
  to per-pair n if needed. **To try:** run our own Cicero on a TF (generates
  `n_metacell`), check the `significance: n_eff=… FDR<=0.05: X/Y` log line.
- **Files:** `Cicero/scripts/03_run_cicero.R`,
  `cobinding/scripts/cicero_conns_to_edges.py`.

### 2026-09-13 — Start of this research journal
- **Context:** Project has grown across many pipelines and several subtle,
  hard-won methods decisions; needed a single reviewable record of major updates.
- **Change:** Created `RESEARCH_JOURNAL.md` at the repo root, back-filled with the
  full history below.
- **Outcome:** Going forward, every major update gets an entry here.

### 2026-09-09 → 09-13 — ChromHMM-18 RPKM correlation: "all-red" was a color-scale artifact
- **Context:** Replicating the reference "TF chromHMM18 rpkm ratio pearson cor"
  figure. First cluster run looked degenerate — a near-uniform red heatmap
  (median off-diagonal Pearson r ≈ 0.96; `r(CTCF,RBBP4)=1.000`).
- **Investigation:** Detoured through "needs an across-TF ratio / log / Spearman"
  (all wrong for this data). The real issue: the meaningful structure lives in the
  0.65–0.90 band, and the original plot auto-scaled `vmin`≈0.68 **and** crammed in
  a 4th panel, saturating everything ≥0.85 to red.
- **Change:** Plain **RPKM → Pearson** is the correct and sufficient pipeline.
  Fixed the plot to the reference's scale (`vmin=0.65, vmax=1.0`, RdYlBu_r) and
  clean layout (dendrogram | heatmap | cell-count bars). With that, the real data
  reproduces the reference's two-block structure — CTCF/STAG2/E2F3/NEUROD4/RBBP4
  vs. MEF2A/THAP6/ZNF143 — faithfully.
- **Outcome:** RPKM normalizes state length + per-TF depth at once; no second
  normalization needed. Lesson: verify the *visualization* before re-deriving the
  method. Also applied the validated ChromHMM RPKM workflow to A/B compartments,
  and recorded why RPKM cannot rescue per-domain compartment correlation.
- **Files:** `tf_complex/scripts/{build_chromhmm_matrix,chromhmm_correlation}.py`,
  `tf_complex/slurm/04_chromhmm_rpkm.sbatch`. **Data:** ENCODE `ENCFF071AXS`
  (HEK293T ChromHMM-18, hg38). **Commits:** `5e8e3f0`, `0685fda`, `45356fa`,
  `afe584d`, `9f8ec27`.

### 2026-09-04 → 09-10 — Motif enrichment: scoring, backgrounds, and the accessibility wash
- **Context:** Can imputed data recover TF-specific motifs, and does an
  open-chromatin-region (OCR) background beat a whole-genome background?
- **Findings / changes:**
  - **AME `--scoring max` vs `avg`.** `max` maximized raw sensitivity but wrecked
    specificity (NRF1's own motif fell from rank 1 → rank 79) and produced
    meaningless 1e-100…1e-4000 p-values (AME's Fisher/rank-sum scales ~exponentially
    with sequence count). Reverted default to **`avg`**, which ranks the target
    motif at the top. The metric that matters is **target-motif rank**, not the
    absolute p-value. (Initial `avg→max` change was commit `7180509`; reverted
    `b220cf0`.)
  - **HOMER OCR background works correctly.** Against "open chromatin minus the
    TF's own peaks," the imputed TF motif comes out P≈1.0 — *real depletion*, not a
    bug: e.g. CTCF motif in only 2.3% of imputed target peaks vs 9.8% of the OCR
    background (real CTCF ChIP is >70%). The imputed peaks land in generic
    accessible regions and miss true sites, which stay in the background. Added
    `-cpg -olen 3` (HOMER practicalTips) to kill poly-G/Maz junk (`b365709`).
  - **Accessibility wash (core biological conclusion).** Imputed q<0.05 peaks are
    largely TF-nonspecific (75–91% shared across TFs) → generic GC motifs; a
    matched OCR background does **not** rescue TF specificity. Honest table:
    Raw recovers the motif; Imputed/genome-bg still recovers it (easy contrast);
    Imputed/OCR-bg fails for both AME(avg) and HOMER → imputation adds
    accessibility, not TF specificity.
  - Consensus subtraction (`TF_SPECIFIC=1`) partially recovers some motifs
    (NRF1 q≈0.025, HOXD10, FOXK1, HOXA5).
- **Files:** `motif_analysis/slurm/motif_enrichment.sbatch`,
  `motif_analysis/summarize_motif_table.py`, `motif_analysis/scripts/emit_bins_bed.py`.
- **Memory:** `imputed-peaks-accessibility-wash.md`.

### 2026-09-04 — TF co-occupancy correlation confound (CPM / fraction is a Pearson no-op)
- **Context:** TF×TF co-occupancy matrix came out as an all-red "rich club";
  attempts to normalize by CPM / per-TF fraction didn't help.
- **Change / outcome:** Proved **per-TF (per-column) scaling — CPM, fraction — is a
  mathematical no-op for Pearson** (scale invariance; max|ΔR|≈2e-16). The confound
  is *structural* (peak-count / domain-size / domain-activity driven), not
  magnitude. Fixes that actually work: correlate over uniform units and normalize
  by both length and depth (→ this fed the ChromHMM-RPKM approach). Removed the
  broken `pca` metric (row-centering made sparse TFs collinear → spurious 36-TF
  block); defaulted to Pearson/Jaccard.
- **Commits:** `4b66def` (PCA bug), `fd811d6`, `8604de8` (CPM), `614afe6`
  (domain size), `b95e027` (fraction), `9970586` (bin width).
- **Memory:** `tf-complex-parallel-analysis.md`.

### 2026-08-26 → 09-01 — A/B compartments + Hi-C, and A-A/B-B co-binding
- **Context:** Stratify TF co-occupancy by A/B compartment to find complexes.
- **Change:** Built the Hi-C pipeline for A/B compartments (matched HEK293T Hi-C,
  SRR24709565, MboI, GSE233166 → hg38 via runHiC → cooltools), set resolution to
  **25 kb**, and added A-A and B-B correlations plus per-compartment motif analysis.
  Reformatted all correlation matrices to the "complexity" layout (heatmap +
  per-TF cell-count bar).
- **Commits:** `c4620a7`, `5459580`, `0d40691`, `ad4e496`, `5c2cba9`, `a7156fb`.
- **Memory:** `hic-compartments-cobinding.md`.

### 2026-08-26 → 09-09 — Co-binding higher-order clusters (Cicero) module
- **Context:** Discover higher-order co-binding regions beyond pairwise.
- **Change:** `cobinding/` module — per-TF Cicero peak co-accessibility → region
  graph → higher-order clusters/cliques (nodes are **peaks**, not TFs). Added
  single force-directed canvas, dropped isolated pairs, and annotated cliques with
  ChromHMM state, A/B compartment, and central genes (30.7% of distal partners sit
  in promoter chromatin).
- **Commits:** `c51d719`, `f8e9ea1`, `c6d9ada`, `46a63e6`, `b3ab734`, `db988eb`,
  `dcf5465`.
- **Memory:** `cobinding-higher-order-clusters.md`.

### 2026-08-19 → 08-25 — Motif Enrichment Analysis (MEA) pipeline introduced
- **Context:** First motif-enrichment capability on the imputed calls.
- **Change:** Added MEA scripts (HOMER known/de-novo + MEME-suite AME over
  HOCOMOCO v11 full), fixed region-shortage and background bugs, wired the
  genome-vs-OCR background comparison, and standardized peaks at MACS3 q<0.05.
- **Commits:** `1ca2f2b`, `acd8b4f`, `8899ec6`, `df9e7be`, `d51dceb`, `e48a65f`.

### 2026-08-17 — Downstream redesign: pos/neg universe, OCR filtering, peak-call standard
- **Context:** Earlier universes/peak definitions were inconsistent across TFs.
- **Change:** Redefined positive/negative universes, added Open-Chromatin-Region
  filtering, and set the project peak-calling standard. Corrected CTCF's positive
  peak set (an inflated ~322k-peak set from the wrong BED → ENCODE `ENCFF314ZAL`,
  HEK293 CTCF ChIP) and the CTCF negative-bin definition (not-CTCF-bound cCRE
  universe + zero-bulk-read filter). Committed configs for all 1000-TF imputations.
- **Commits:** `8f61059`, `dfad51a`, `a3b3770`, `d8d0ffa`, `9a2b847`.
- **Memory:** `ctcf-positive-peakset-fix.md`, `ctcf-negative-bin-definition.md`.

### 2026-07 → 2026-08-17 — Multi-TF downstream expansion + open-chromatin masking
- **Context:** Scale downstream from a few TFs to the panel; make imputed-only
  calls outside accessible chromatin less credible.
- **Change (detailed in `CHANGES_July_to_August_2026.md`):** `all_universe` eval
  mode and bulk×bulk reference matrices; HEK293T OmniATAC open-chromatin mask
  (GSE302716 via ENCODE ATAC pipeline v2.2.3; IDR-optimal ~72,182 peaks →
  ~57,286/3.03M bins "open"); post-mask peak recall and refreshed slides; large
  disk cleanup (~187 GB → ~11 GB) and `.gitignore` overhaul.
- **Note:** Most of this period was **not** committed (git quiet 07-01 → 08-17);
  that markdown is the durable record.

### 2026-06-04 → 06-29 — Unified imputer + peak-calling standardization; TF panel growth
- **Context:** Consolidate multiple imputation methods into one model and settle a
  reproducible peak-calling scheme.
- **Change:** Added the **unified** model; migrated MACS2 → **MACS3**; swept
  peak-calling q-value / span / per-bin quantile and coverage-spread parameters;
  tightened negative-background definition; added data-preparation pipelines and
  configs for new TFs with bulk data.
- **Commits:** `88b0e99`, `be1fc07`, `b558aed`, `641427a`, `e807…`(data-prep),
  `d4ab8f6`.

### 2026-05 — Imputation method bake-off
- **Context:** Find the best imputer for sparse per-TF binding.
- **Change:** Implemented and compared **cisTopic, FITS, scOpen, MAGIC, PUscOpen,
  scBasset, Borzoi**, plus **combined** pipelines (scBasset×cisTopic,
  scBasset×PUscOpen, scBasset_cisTopic_PUscOpen). Added Sens/Spec/F1 and
  held-out-reconstruction evals, bulk-comparison metrics, Cicero, and bigwig
  export. Repeatedly fixed data-leakage in the PU (positive-unlabeled) masking.
- **Commits:** many; e.g. `f28974e` (cisTopic), `911600a` (FITS), `db81a66`
  (scOpen), `102da25` (MAGIC), `a2e9757` (PUscOpen), `5d23e57` (scBasset),
  `7cc3a68` (Borzoi), `0d46f1f`/`fe7318d`/`224c91f` (leakage fixes).

### 2026-04-17 → 04-25 — Project bootstrap
- **Context:** Stand up the repo and the first imputation pipeline.
- **Change:** Initial import of the Paired-Tag pipeline, references, and conda env;
  renamed scope to **XuLab-Computations**; first cisTopic TF-imputation pipeline
  with data loaders, sparsity-tuned configs, and a held-out reconstruction eval.
- **Commits:** `b37d1a3`, `a5e2697`, `f28974e`, `86d7674`.

---

## Standing lessons (quick reference)
- **Cluster sync:** unchanged output usually = stale checkout on Longleaf; pull first.
- **Pearson scale-invariance:** per-TF CPM/fraction can't de-confound a Pearson
  correlation; fix the *units* (length + depth), not the magnitude.
- **Imputation = accessibility, not specificity:** imputed q<0.05 peaks are 75–91%
  shared across TFs; motif specificity survives only vs. easy backgrounds.
- **Motif ranking > motif p-value:** at large N every real motif is "1e-huge"; use
  the target's rank and its gap over the generic-GC pack.
- **Check the plot before the method:** the ChromHMM "all-red" scare was a `vmin`
  color-scale artifact, not a broken computation.
