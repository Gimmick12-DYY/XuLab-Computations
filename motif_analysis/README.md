# Motif analysis

Motif enrichment, de-novo discovery, and novel-motif finding for the HEK293T
scCUT&Tag TF panel. Moved out of `downstream/` so peak-coverage / slide tables
stay separate from motif result trees.

## Layout

```
motif_analysis/
  scripts/          # enrichment helpers, Codebook fetch, ranking, panel builders
  slurm/            # motif_enrichment, denovo_{pmat,imputed,specific}, rank_motifs
  cache/motifdb/    # HOCOMOCO v12 / JASPAR / Codebook / merged MEME DBs (gitignored)
  motif/            # imputed-peak enrichment (AME + HOMER known/de-novo)
  motif_bins/       # same pipeline on top binding bins
  motif_tfspec/     # consensus-subtracted peaks
  motif_pmat/       # de-novo on fragment-called pmat peaks (unknown TFs)
  motif_imputed/    # same de-novo protocol on top imputed peaks
  motif_ranked/     # Step-3 performance ranking outputs (optional)
  MOTIF_ENRICHMENT.md
  NOVEL_MOTIF_FINDING.md
  motif_reference_manifest.tsv
  summarize_motif_table.py
```

Shared utilities that stay in `downstream/`:
`call_imputed_peaks.py`, `peak_coverage.py`, `cache/hg38.fa`, HOCOMOCO v11 full MEME.

## Quick start

```bash
# Known + de-novo enrichment on imputed peaks
LIST=1 bash motif_analysis/slurm/motif_enrichment.sbatch
N=$(LIST=1 bash motif_analysis/slurm/motif_enrichment.sbatch | grep -vc '^#')
sbatch --array=0-$((N-1)) motif_analysis/slurm/motif_enrichment.sbatch

# De-novo on pmat peaks (unknown TFs)
sbatch --array=0-44 motif_analysis/slurm/motif_denovo_pmat.sbatch

# Codebook (Zenodo) + merge into TOMTOM DB
bash motif_analysis/scripts/fetch_codebook.sh
python motif_analysis/scripts/merge_meme_db.py \
  --in motif_analysis/cache/motifdb/H12CORE_meme_format.meme=h12 \
       motif_analysis/cache/motifdb/codebook_top1.meme=cdbk \
  --out motif_analysis/cache/motifdb/merged_human_motifs.meme
```

See `MOTIF_ENRICHMENT.md` and `NOVEL_MOTIF_FINDING.md` for full details.
