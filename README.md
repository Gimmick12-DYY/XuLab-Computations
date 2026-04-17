# Xu Lab — Paired data analysis

This repository collects **Paired-Tag / paired multimodal** analysis assets used in the Xu lab: preprocessing and mapping scripts, `reachtools`, reference files, and downstream R workflows.

The core pipeline and documentation live under [`Paired-Tag/`](Paired-Tag/).

## Repository layout

| Path | Purpose |
|------|---------|
| [`environment.yml`](environment.yml) | Conda environment with common aligners and QC tools |
| [`Paired-Tag/README.md`](Paired-Tag/README.md) | End-to-end preprocessing: barcode extraction, DNA/RNA mapping, matrix merge |
| [`Paired-Tag/pipeline/readme.md`](Paired-Tag/pipeline/readme.md) | Wrapper pipeline (`run.sh`) and file naming conventions |
| [`Paired-Tag/protocol/readme.md`](Paired-Tag/protocol/readme.md) | Protocol PDF link and wet-lab FAQs |
| [`Paired-Tag/reachtools/`](Paired-Tag/reachtools/) | C++ utilities; build with `sh make.sh` after editing paths |
| [`Paired-Tag/shellscrips/`](Paired-Tag/shellscrips/) | Shell drivers for FASTQ preprocessing and genome alignment |
| [`Paired-Tag/perlscripts/`](Paired-Tag/perlscripts/) | Matrix filter/merge and BAM helpers |
| [`Paired-Tag/rscripts/`](Paired-Tag/rscripts/) | QC plots, Seurat, and integration examples |
| [`Paired-Tag/refereces/`](Paired-Tag/refereces/) | Cellular barcode FASTA/Bowtie indexes and RNA/bin annotation lists |

## Environment

Create the conda environment from the repo root:

```bash
conda env create -f environment.yml
conda activate paired-tag
```

The environment includes Bowtie, Bowtie2, STAR, Trim Galore, Samtools, FastQC, Perl, and Make. Build **reachtools** and Bowtie indexes on barcode FASTA as described in [`Paired-Tag/README.md`](Paired-Tag/README.md).

## Quick workflow

1. **Build tools and references** — `reachtools` + Bowtie index on `cell_id_full.fa` or `cell_id_full_407.fa` (see Paired-Tag README).
2. **Preprocess FASTQs** — [`Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh`](Paired-Tag/shellscrips/01.pre_process_paired_tag_fastq.sh) (adjust paths; note Bowtie 0.x vs 1.x and GEO/SRA read-name caveats in script comments).
3. **Map** — DNA: [`02.proc_DNA.sh`](Paired-Tag/shellscrips/02.proc_DNA.sh); RNA: [`03.proc_RNA.sh`](Paired-Tag/shellscrips/03.proc_RNA.sh).
4. **Merge and analyze** — filter low-read barcodes if desired, merge sub-libraries with `perlscripts/merge_mtx.pl`, then use R/Seurat or other tools as in the Paired-Tag README.

For a single entry point with fixed paths, see [`Paired-Tag/pipeline/run.sh`](Paired-Tag/pipeline/run.sh) and its readme.

## License and attribution

Pipeline code and documentation in `Paired-Tag/` follow the upstream [**Paired-Tag**](https://github.com/cxzhu/Paired-Tag) project; see [`Paired-Tag/LICENSE`](Paired-Tag/LICENSE).

If you use this method in a publication, cite the Paired-Tag paper:

> Zhu *et al.*, Joint profiling of histone modifications and transcriptome in single cells from mouse brain. *Nature Methods* (2021). [https://doi.org/10.1038/s41592-021-01060-3](https://doi.org/10.1038/s41592-021-01060-3)
