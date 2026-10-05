# Xu Lab — Computations

**XuLab-Computations** is the Xu lab’s shared repository for computational workflows and tooling. It currently includes:

1. **Paired-Tag / paired multimodal** — preprocessing and mapping scripts, `reachtools`, reference files, downstream R workflows, and pileup utilities.
2. **cisTopic** — a standalone pipeline for **single-cell TF / accessibility matrices** using [pycisTopic](https://github.com/aertslab/pycisTopic): export from `.rds`, LDA with MALLET, model selection, and imputation (`theta` / `phi` or full `P(r|c)`).
3. **unified** — a single-architecture PyTorch imputer for sparse TF / chromatin matrices: Basset-style sequence encoder, multi-scale bin-context conv, learnable cell bank, and a gated bilinear head, trained jointly (focal + ranking loss). Share pack (code only): [`share/unified_imputation_pipeline.zip`](share/unified_imputation_pipeline.zip).

Additional pipelines may live alongside this tree over time.

## Repository layout

| Path | Purpose |
|------|---------|
| [`README.md`](README.md) | This top-level guide for repository structure, setup, and quick-start workflows |
| [`.gitignore`](.gitignore) | Ignore rules for generated data and local tooling artifacts |
| [`environment.yml`](environment.yml) | Conda env **`paired-tag`**: aligners and QC for Paired-Tag (Bowtie, Bowtie2, STAR, Trim Galore, Samtools, FastQC, Perl, Make) |
| [`Paired-Tag/README.md`](Paired-Tag/README.md) | End-to-end Paired-Tag preprocessing: barcode extraction, DNA/RNA mapping, matrix merge |
| [`Paired-Tag/pipeline/readme.md`](Paired-Tag/pipeline/readme.md) | Wrapper pipeline (`run.sh`) and file naming conventions |
| [`Paired-Tag/protocol/readme.md`](Paired-Tag/protocol/readme.md) | Protocol PDF link and wet-lab FAQs |
| [`Paired-Tag/reachtools/`](Paired-Tag/reachtools/) | C++ utilities; build with `sh make.sh` after editing paths |
| [`Paired-Tag/shellscrips/`](Paired-Tag/shellscrips/) | Shell drivers for FASTQ preprocessing and genome alignment |
| [`Paired-Tag/perlscripts/`](Paired-Tag/perlscripts/) | Matrix filter/merge and BAM helpers |
| [`Paired-Tag/rscripts/`](Paired-Tag/rscripts/) | QC plots, Seurat, and integration examples |
| [`Paired-Tag/refereces/`](Paired-Tag/refereces/) | Cellular barcode FASTA/Bowtie indexes and RNA/bin annotation lists |
| [`Paired-Tag/remove_pileup/`](Paired-Tag/remove_pileup/) | Scripts to count/remove pileups from BAM-derived data (`remove_pileups.py`, `run.sh`) |
| [`cisTopic/README.md`](cisTopic/README.md) | cisTopic LDA + imputation pipeline details, data scale notes, and references |
| [`cisTopic/environment.yml`](cisTopic/environment.yml) | Conda env **`cistopic`** (Python/R stack; `pycisTopic` installed from GitHub via pip) |
| [`cisTopic/configs/default.yaml`](cisTopic/configs/default.yaml) | Main runtime configuration (input/work paths, MALLET path, filtering, LDA grid, imputation mode) |
| [`cisTopic/scripts/`](cisTopic/scripts/) | Pipeline scripts `00`-`07`: inspect/export/build/LDA/select/impute/downstream/eval |
| [`cisTopic/slurm/`](cisTopic/slurm/) | SLURM templates to submit each cisTopic stage and chained dependencies |
| [`cisTopic/scripts/cistopic_ctcf/`](cisTopic/scripts/cistopic_ctcf/) | Example working directory with generated run outputs (`mm`, `obj`, `models`, `select`, `impute`, `downstream`, `eval`) |
| [`Mallet-202108/`](Mallet-202108/) | Local MALLET distribution used by cisTopic (`paths.mallet_path`) |
| [`unified/README.md`](unified/README.md) | Full unified docs: model structure, required inputs, config, manual/SLURM run |
| [`unified/environment.yml`](unified/environment.yml) | Conda env **`unified`** (Python 3.11 + PyTorch + einops) |
| [`unified/configs/`](unified/configs/) | `default.yaml` + per-TF overlays (`paths`, `model`, `train`, `impute`) |
| [`unified/scripts/`](unified/scripts/) | `00_ingest` → `01_train` → `02_impute` (+ `02b` remask, `03` RDS export); `_model.py` |
| [`unified/slurm/`](unified/slurm/) | Full-pipeline and multi-TF SLURM drivers (edit paths/partitions for your cluster) |
| [`share/unified_imputation_pipeline.zip`](share/unified_imputation_pipeline.zip) | Share pack: code + configs only — **no** FASTA, BEDs, matrices, or `work/` |

Folder names `shellscrips` and `refereces` match the upstream Paired-Tag layout.

## Environments

**Paired-Tag** (repo root):

```bash
conda env create -f environment.yml
conda activate paired-tag
```

**cisTopic** uses a separate environment (pycisTopic, R for export/inspect):

```bash
conda env create -f cisTopic/environment.yml
conda activate cistopic
```

**unified** (PyTorch imputer; keep isolated from cisTopic):

```bash
conda env create -f unified/environment.yml
conda activate unified
```

Notes:
- `pycisTopic` is installed from GitHub in `cisTopic/environment.yml` (not from PyPI).
- `paths.mallet_path` in `cisTopic/configs/default.yaml` must point to a real MALLET binary, e.g. `Mallet-202108/bin/mallet`.
- The unified share zip does **not** include a genome FASTA or ATAC/blacklist BEDs. Recipients set `paths.genome_fa`, supply `mm/` (Matrix Market + regions + barcodes), and should set `impute.open_chromatin_bed: null` unless they provide their own open-chromatin BED.

See [`cisTopic/README.md`](cisTopic/README.md) and especially [`unified/README.md`](unified/README.md) for architecture and end-to-end instructions. Also [`ENVIRONMENTS.md`](ENVIRONMENTS.md).

## Paired-Tag quick workflow

1. **Build tools and references** — `reachtools` + Bowtie index on `cell_id_full.fa` or `cell_id_full_407.fa` (see [`Paired-Tag/README.md`](Paired-Tag/README.md)).
2. **Preprocess FASTQs** — [`Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh`](Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh) (adjust paths; note Bowtie 0.x vs 1.x and GEO/SRA read-name caveats in script comments).
3. **Map** — DNA: [`02.proc_DNA.sh`](Paired-Tag/shellscrips/02.proc_DNA.sh); RNA: [`03.proc_RNA.sh`](Paired-Tag/shellscrips/03.proc_RNA.sh).
4. **Merge and analyze** — filter low-read barcodes if desired, merge sub-libraries with `perlscripts/merge_mtx.pl`, then use R/Seurat or other tools as in the Paired-Tag README.

For a single entry point with fixed paths, see [`Paired-Tag/pipeline/run.sh`](Paired-Tag/pipeline/run.sh) and its readme.

## cisTopic quick pointer

High level: inspect `.rds` → export Matrix Market → build CistopicObject → MALLET LDA over a topic grid → select topic count K → save `theta`/`phi` (and optionally full imputed `P(r|c)`) → optional downstream UMAP/topic binarization.

The current workflow in this repo has been run end-to-end (`02` to `06`) and now includes a held-out reconstruction benchmark:

```bash
conda run -n cistopic python cisTopic/scripts/07_eval_heldout.py \
  --config cisTopic/configs/default.yaml \
  --n-samples 100000 --seed 42
```

This writes AUROC/AUPRC benchmarking outputs under `<work_dir>/eval/`:
- `heldout_eval.json`
- `heldout_eval.tsv`
- `heldout_eval.png`

You can also run strict held-out mode by first creating a masked matrix with `--prepare-holdout`, then retraining and scoring with `--holdout-split` (see script header in `cisTopic/scripts/07_eval_heldout.py`).

Run locally or chain SLURM jobs under [`cisTopic/slurm/`](cisTopic/slurm/). Full steps, storage notes, and citations are in [`cisTopic/README.md`](cisTopic/README.md).

## unified quick pointer

**Model (one jointly trained net):** DNA window → SequenceEncoder (`D=64`) → BinContext (dilations 1/4/16 along chrom bins) → bilinear cell bank `p_rc` × GateMLP `g_rc` → \(\hat y = g\cdot p\). Trained with focal BCE + uniform negatives + pairwise ranking; optional RC augmentation. Open-chromatin BED filtering is **post-hoc and optional**, not part of the network.

**You provide:** (1) sparse counts as `mm/{matrix.mtx.gz, regions, barcodes}`, (2) indexed genome FASTA matching contig names, (3) optional blacklist / open-chromatin BEDs. The zip ships none of these.

```bash
unzip share/unified_imputation_pipeline.zip -d /path/to/dest
cd /path/to/dest/unified

conda env create -f environment.yml && conda activate unified

# Edit configs/default.yaml:
#   paths.work_dir, paths.genome_fa
#   impute.open_chromatin_bed: null   # unless you supply a BED
# Place Matrix Market files under $work_dir/mm/

python scripts/00_ingest.py --config configs/default.yaml
python scripts/01_train.py  --config configs/default.yaml   # GPU
python scripts/02_impute.py --config configs/default.yaml   # GPU
# outputs: $work_dir/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json}
```

Full submodule dimensions, loss details, work-dir layout, and SLURM notes: [`unified/README.md`](unified/README.md).

## License and attribution

Pipeline code and documentation in `Paired-Tag/` follow the upstream [**Paired-Tag**](https://github.com/cxzhu/Paired-Tag) project; see [`Paired-Tag/LICENSE`](Paired-Tag/LICENSE).

If you use Paired-Tag in a publication, cite:

> Zhu *et al.*, Joint profiling of histone modifications and transcriptome in single cells from mouse brain. *Nature Methods* (2021). [https://doi.org/10.1038/s41592-021-01060-3](https://doi.org/10.1038/s41592-021-01060-3)

For cisTopic methods, cite cisTopic / pycisTopic as in [`cisTopic/README.md`](cisTopic/README.md).
