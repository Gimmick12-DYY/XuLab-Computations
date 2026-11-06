# Composite-motif pair test (pair pool vs single-TF pools)

Question: do bins bound by both TFs of a pair carry a motif that neither TF's own bins carry?

## Pools (`build_pools.py`)
- Per TF: union-ATAC 1 kb bins (104,738) ranked by imputed CPM minus the 79-TF mean. The signal matrix is `../motif_shared_20261103/fg/signal_mask.npz` (from `unified/work/<tf>/impute`).
- Bins dropped before ranking: ENCODE blacklist v2 overlap, >50% exonic (GENCODE v46 basic), N in the centre 200 bp. 95,044 bins remain; each TF keeps its top 10,000.
- Pair A-B: `pair` = in both top sets; `Aonly` and `Bonly`; `input` = eligible bins in neither, matched to the pair pool on GC (0.02 steps) x CpG o/e (quintiles), 2 per pair bin where available; `singles` = Aonly + Bonly.
- Sequences: bin centre +/- 100 bp, hg38. Discovery = odd autosomes (`disc/`), validation = even autosomes + chrX (`val/`).
- `pool_sizes.tsv`: a pair with fewer than 300 discovery pair bins is untestable.
- `pairs.tsv` lists the gate pairs, the negative (RBBP7-SRCAP), 20 random pairs (4 per gate/negative, drawn from the 30 pairs closest in top-set overlap) and the candidates (AF3 strong+weak from `../af3_novel_20261103`, plus the named pairs).
- `--perms`: 50 label permutations per gate/negative pair. Pair, A-only and B-only labels are reshuffled within A u B, within strata of expected reads (79-TF mean CPM quintile) x GC.

## Motif runs (`run_pair.sbatch`, task files from `tasks.py`)
- HOMER 5.1 `findMotifsGenome.pl hg38 -size given -p 1`, run pair vs A-only, vs B-only and vs input. Each contrast is 4 tasks: known only (`-nomotif`), and de novo at length 8, 10 or 12 (`-noknown`).
- MEME 5.5.7:
  - `sea` and `centrimo`: pair vs singles, against `db/known.meme` (JASPAR 2024 CORE vertebrates non-redundant + HOCOMOCO v12 CORE + Codebook top-1 + the CAP-SELEX MLX_FOXA1 / MLX_FOXD2 composites; built by `build_db.sh` with `meme2meme`).
  - `spamo`: own motif as primary (CTCF MA0139.2; MAZ MA1522.2 and ZNF143 MA0088.2; NFYA MA0060.4).
  - `streme` pair vs singles, then `tomtom` against the DB.
- `perm`: HOMER known only, permuted pair vs permuted A-only / B-only.
- `validate`: candidate motifs on the even-chromosome pools, pair vs singles. HOMER `-mknown` for HOMER candidates; `sea` for STREME candidates.

## Calls (`summarize.py`)
- Candidate: present in both the vs-A and vs-B runs. Known motifs must be the same HOMER motif; de novo motifs must have column-Pearson similarity >= 0.6 (HOMER-style, padded alignment). Fold >= 1.5 over each single pool and p <= 1e-5 in both runs.
- Gate pass: the gate pair's own known motif is a candidate (CTCF; MAZ and/or ZNF143; NFY; E2F3).
- Permutation p: (1 + #perms whose best candidate p <= observed) / 51.
- Validation: log2FC > 0.75 and P < 0.01 on the even chromosomes.

Outputs (`pools/`, `results/`, `candidates/`, `ref/`, `db/`, `logs/`, `preparsed/`, tables) are not tracked.
