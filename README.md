# Xu Lab — Computations

**XuLab-Computations** is the Xu lab’s shared repository for computational workflows and tooling. It currently includes:

1. **Paired-Tag / paired multimodal** — preprocessing and mapping scripts, `reachtools`, reference files, downstream R workflows, and pileup utilities.
2. **cisTopic** — a standalone pipeline for **single-cell TF / accessibility matrices** using [pycisTopic](https://github.com/aertslab/pycisTopic): export from `.rds`, LDA with MALLET, model selection, and imputation (`theta` / `phi` or full `P(r|c)`).

Additional pipelines may live alongside this tree over time.

## Repository layout

| Path | Purpose |
|------|---------|
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
| [`cisTopic/README.md`](cisTopic/README.md) | cisTopic LDA + imputation pipeline; own conda env, `configs/default.yaml`, Python/R scripts, SLURM `sbatch` templates |

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

See [`cisTopic/README.md`](cisTopic/README.md) for MALLET setup and `configs/default.yaml`.

## Paired-Tag quick workflow

1. **Build tools and references** — `reachtools` + Bowtie index on `cell_id_full.fa` or `cell_id_full_407.fa` (see [`Paired-Tag/README.md`](Paired-Tag/README.md)).
2. **Preprocess FASTQs** — [`Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh`](Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh) (adjust paths; note Bowtie 0.x vs 1.x and GEO/SRA read-name caveats in script comments).
3. **Map** — DNA: [`02.proc_DNA.sh`](Paired-Tag/shellscrips/02.proc_DNA.sh); RNA: [`03.proc_RNA.sh`](Paired-Tag/shellscrips/03.proc_RNA.sh).
4. **Merge and analyze** — filter low-read barcodes if desired, merge sub-libraries with `perlscripts/merge_mtx.pl`, then use R/Seurat or other tools as in the Paired-Tag README.

For a single entry point with fixed paths, see [`Paired-Tag/pipeline/run.sh`](Paired-Tag/pipeline/run.sh) and its readme.

## cisTopic quick pointer

High level: inspect `.rds` → export Matrix Market → build CistopicObject → MALLET LDA over a topic grid → select topic count K → save `theta`/`phi` (and optionally full imputed `P(r|c)`). Run locally or chain SLURM jobs under [`cisTopic/slurm/`](cisTopic/slurm/). Full steps, storage notes, and citations are in [`cisTopic/README.md`](cisTopic/README.md).

## License and attribution

Pipeline code and documentation in `Paired-Tag/` follow the upstream [**Paired-Tag**](https://github.com/cxzhu/Paired-Tag) project; see [`Paired-Tag/LICENSE`](Paired-Tag/LICENSE).

If you use Paired-Tag in a publication, cite:

> Zhu *et al.*, Joint profiling of histone modifications and transcriptome in single cells from mouse brain. *Nature Methods* (2021). [https://doi.org/10.1038/s41592-021-01060-3](https://doi.org/10.1038/s41592-021-01060-3)

For cisTopic methods, cite cisTopic / pycisTopic as in [`cisTopic/README.md`](cisTopic/README.md).
