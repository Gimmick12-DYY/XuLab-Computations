# Xu Lab — Computations

**XuLab-Computations** is the Xu lab’s shared repository for computational workflows and tooling. It currently includes:

1. **Paired-Tag / paired multimodal** — preprocessing and mapping scripts, `reachtools`, reference files, downstream R workflows, and pileup utilities.
2. **cisTopic** — a standalone pipeline for **single-cell TF / accessibility matrices** using [pycisTopic](https://github.com/aertslab/pycisTopic): export from `.rds`, LDA with MALLET, model selection, and imputation (`theta` / `phi` or full `P(r|c)`).

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

Notes:
- `pycisTopic` is installed from GitHub in `cisTopic/environment.yml` (not from PyPI).
- `paths.mallet_path` in `cisTopic/configs/default.yaml` must point to a real MALLET binary, e.g. `Mallet-202108/bin/mallet`.

See [`cisTopic/README.md`](cisTopic/README.md) for full setup details.

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

## License and attribution

Pipeline code and documentation in `Paired-Tag/` follow the upstream [**Paired-Tag**](https://github.com/cxzhu/Paired-Tag) project; see [`Paired-Tag/LICENSE`](Paired-Tag/LICENSE).

If you use Paired-Tag in a publication, cite:

> Zhu *et al.*, Joint profiling of histone modifications and transcriptome in single cells from mouse brain. *Nature Methods* (2021). [https://doi.org/10.1038/s41592-021-01060-3](https://doi.org/10.1038/s41592-021-01060-3)

For cisTopic methods, cite cisTopic / pycisTopic as in [`cisTopic/README.md`](cisTopic/README.md).
