# Higher-order co-binding clusters (per TF)

From a TF's pairwise Cicero co-accessibility (proximal/distal peak pairs) to
**higher-order clusters of co-binding regions** — sets of that TF's binding sites
that mutually co-bind (the "if a–b, b–c, a–c, they're a hub" question).

This is **per TF** (RBBP4 shown): nodes are **peaks/regions**, edges are
co-accessible pairs, clusters are groups of RBBP4 regions that co-bind together.

## Input: Cicero connection files (per TF)

`data/TF.<TF>.fitConns.res.{sel,pdc}`:
- **`.sel`** — the full selected network: distal–distal + proximal–proximal +
  proximal–distal. **Non-bipartite → true a–b–c cliques form.** Use this.
- **`.pdc`** — proximal–distal subset (distal annotated by chromatin state).
  Proximal–distal only → **bipartite → no triangles/cliques** (community
  detection still works).

Parsing is **layout-agnostic**: each row's two genomic coordinates become the two
peak nodes (id `chr_s_e` and coord `chr:s-e` both normalized), coaccess/q come
from the trailing numeric columns, and peak type (proximal/distal) + annotation
(gene / chromatin state = the token after the type) are captured best-effort.

## Pipeline

```bash
mamba env create -f cobinding/environment.yml -p /work/users/d/y/dyy12/conda/envs/cobinding

sbatch cobinding/slurm/run_cobinding.sbatch                    # TF=RBBP4, all files (.sel+.pdc)
TF=RBBP4 QVAL=0.01 sbatch cobinding/slurm/run_cobinding.sbatch
REBUILD=0 QVAL=0.01 sbatch cobinding/slurm/run_cobinding.sbatch   # re-cluster only

# full-network only (skip .pdc): GLOB='TF.{tf}.fitConns.res.sel'
```

### 1. `build_peak_graph.py` — peak graph (run once per TF)
Globs the TF's connection files, normalizes coordinates, dedups peak-pair edges
(max coaccess, min q). Writes `work/<tf>.peak_edges.tsv`
(`peak1 peak2 coaccess pval qval type1 type2 n_links`) and `work/<tf>.peak_annot.tsv`
(`peak types genes states n_edges`).

### 2. `cluster_peaks.py` — cliques + overlap modules (fast; tune here)
Filters edges (FDR ≤ `QVAL`), builds the peak graph, and writes to `results/<tf>/`:
- **`cliques.tsv`** — **strict maximal cliques** (≥ `MIN_CLIQUE`): every region
  co-binds every other (`density = 1`). The fully-coordinated cores.
- **`modules.tsv`** — **overlap modules**: strict cliques merged only when they
  **share ≥ `SHARE` regions** (default 2 = an edge). A partially-complete
  higher-order architecture anchored by complete subgraphs. Columns:
  `module_id n_regions n_source_cliques n_edges density chromosome span_bp
  region_types genes … max_pair_fdr states regions`.
- **`nodes.tsv`** — peak, degree, k-core, n_cliques, n_modules, type, genes, states.
- **`summary.txt`** + (`--plot`) `clique_sizes.png`, `module_sizes.png`.

> Reproduces the reference workbook exactly on the full RBBP4 `.sel`
> (FDR≤0.05: 355 cliques / 326 modules; FDR≤0.01: 300 / 277).
> Needs the full `.sel` network for cliques (bipartite `.pdc` yields none).

### Knobs (env vars on the sbatch)

| var | default | meaning |
|-----|---------|---------|
| `TF` | `RBBP4` | which TF's files to process |
| `GLOB` | `TF.{tf}.fitConns.res.*` | input files (`…res.sel` for full-network only) |
| `REBUILD` | `1` | `0` = reuse the built peak graph |
| `QVAL` | `0.05` | edge FDR cutoff (≤); run again at `0.01` for the strict tier |
| `COACCESS_MIN` | `0.0` | min Cicero coaccess per edge |
| `MIN_SUPPORT` | `1` | min `n_links` per peak pair |
| `MIN_CLIQUE` | `3` | smallest clique reported |
| `SHARE` | `2` | cliques merge into a module when sharing ≥ this many regions |

## Reading the result

- **Cliques** = the strict a–b–c sets: every region co-binds every other
  (density 1). The tight, fully-coordinated cores.
- **Overlap modules** = the cluster unit. Cliques that share ≥2 regions are fused
  into a partially-complete higher-order architecture (density < 1) still anchored
  by complete subgraphs. **Focus on `n_source_cliques > 1`** — those are the
  genuine "beyond a single clique" assemblies (on RBBP4: 25 of 326 at FDR≤0.05).
- **`nodes.tsv`**: `n_cliques` / `n_modules` per region flag the shared "hinge"
  regions that anchor higher-order coordination; `k-core` ranks embeddedness.
- Module genomic spans tie naturally into the Hi-C **A/B compartment** work.

## Notes

- **Need the `.sel` file for cliques.** With `.pdc` alone the graph is bipartite
  (proximal↔distal), so `cliques.tsv` is empty by construction — the script says so.
- Distal **chromatin states** populate only if present in the input (the `.pdc`
  here has `nan` in the distal-gene column); point the parser at a file that
  carries the state, or add a distal-state BED annotation step.

## Tracking

`cobinding/work/` and `cobinding/results/` are gitignored (regenerable). Tracked:
`scripts/`, `slurm/`, `environment.yml`, README.
