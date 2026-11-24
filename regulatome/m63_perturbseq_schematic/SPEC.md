# m63: Perturb-seq benchmark slide (data structure schematic)

2026-11-24, user: "Yep I like it, please add it in the local deck" (the approved draft of the X-Atlas/Orion data-structure diagram).

One new slide in ~/Desktop/HEK293T-ATGTAG/293T Yuyang local.pptx, inserted as slide 2 (right after the 11/20/26 Regulome Stage 1 divider); nothing else in the deck changes. Native, editable shapes, no picture; each of the three panels is one group. Title as the deck's Stage 1 slides (Cambria 28 pt bold text box); subtitle grey 14 pt; panel headers Cambria 16 pt bold; labels inside the diagram Arial, all >= 12 pt; footnote Cambria 12 pt grey.

Title: "Perturb-seq benchmark: what the data looks like". Subtitle: "X-Atlas/Orion, genome-wide CRISPRi Perturb-seq in HEK293T (Huang et al., bioRxiv 2025)".
Panel 1 "Raw data: one row per cell": sketch table (guide target | lane | guide UMIs | counts, 38,606 genes), 3 NRF1 rows, CTCF, MAZ, 2 non-targeting; 4.5 million cells in 223 lanes; 18,903 targeted genes; 218,838 control cells.
Panel 2 "For each TF: knockdown vs control": NRF1 343 guide cells, top 25 % by guide UMIs = 86; non-targeting cells of the same lanes = 74,165 (about 1,000 per lane); 63 TFs, median 59 cells per TF (25 to 216).
Panel 3 "Result: TF by gene table": 63 TFs x 11,228 expressed genes; a cutoff gives each TF's Perturb-seq targets (|z| >= 2: median 873 genes per TF).
Footnote: squares, bars and dots are a sketch (fixed seed 20261124), not data values; own-gene knockdown median 63 %.
Numbers are the checked values from m23 / m57 (no recomputation). Builder: m63_deck.py, run by run_deck.sbatch on deck_work/in.pptx -> out.pptx; slide and notes parts renumbered; render of slides 1-3 to PNG for checking.
