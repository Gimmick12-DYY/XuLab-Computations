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
(`peak1 peak2 coaccess qval type1 type2 n_links`) and `work/<tf>.peak_annot.tsv`
(`peak types genes states n_edges`).

### 2. `cluster_peaks.py` — higher-order clusters (fast; tune here)
Filters edges, builds the weighted peak graph, and writes to `results/<tf>/`:
- **`clusters.tsv`** — communities (Louvain / `METHOD=leiden`), each annotated with
  n_proximal / n_distal, gene set, chromatin-state composition, and **genomic span**.
- **`cliques.tsv`** — maximal cliques (fully co-accessible peak sets); needs the
  full `.sel` network (bipartite `.pdc` yields none).
- **`nodes.tsv`** — peak, community, k-core, degree, type, genes, states.
- **`summary.txt`** + (`--plot`) `tf_peak_network.png`, `clique_sizes.png`.

### Knobs (env vars on the sbatch)

| var | default | meaning |
|-----|---------|---------|
| `TF` | `RBBP4` | which TF's files to process |
| `GLOB` | `TF.{tf}.fitConns.res.*` | input files (`…res.sel` for full-network only) |
| `REBUILD` | `1` | `0` = reuse the built peak graph |
| `QVAL` | `0.05` | edge q-value cutoff |
| `COACCESS_MIN` | `0.0` | min Cicero coaccess per edge |
| `MIN_SUPPORT` | `1` | min `n_links` per peak pair |
| `METHOD` | `louvain` | `louvain` \| `leiden` |
| `MIN_CLIQUE` | `3` | smallest clique reported |
| `MIN_CLUSTER` | `3` | smallest community reported |

## Reading the result

- **Clusters** = higher-order co-binding hubs. On RBBP4 the top ones span ~2–3 Mb
  local domains (promoters + enhancers co-binding within a regulatory
  neighbourhood) — a natural tie-in to the Hi-C **A/B compartment** work.
- **Cliques** (with `.sel`) = the strict a–b–c sets: every region co-binds every
  other.
- **k-core** in `nodes.tsv` ranks how embedded a region is in dense co-binding.

## Notes

- **Need the `.sel` file for cliques.** With `.pdc` alone the graph is bipartite
  (proximal↔distal), so `cliques.tsv` is empty by construction — the script says so.
- Distal **chromatin states** populate only if present in the input (the `.pdc`
  here has `nan` in the distal-gene column); point the parser at a file that
  carries the state, or add a distal-state BED annotation step.

## Tracking

`cobinding/work/` and `cobinding/results/` are gitignored (regenerable). Tracked:
`scripts/`, `slurm/`, `environment.yml`, README.
