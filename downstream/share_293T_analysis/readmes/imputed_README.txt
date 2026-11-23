Imputed single-cell TF binding, HEK293T, hg38.

Model: "unified" imputer (XuLab repo, unified/). For each TF it predicts binding per 1 kb bin x cell,
binarized at a per-TF threshold (sparsity_match:3). Every raw call (cell with >= 1 read in the bin) is kept;
the mask only limits where model-added calls are kept.

ATAC_mask/           model-added calls kept only in the 104,738 1 kb bins overlapping HEK293T bulk ATAC peaks
                     (union of three HEK293T ATAC peak sets, HEK293T_ATAC_union3.bed.gz). 79 TFs.
accessibility_mask/  model-added calls kept in 1 kb bins with mean bulk ATAC CPM > 0.0781542 (681,817 bins;
                     the cutoff is the 5th percentile over CTCF consensus peaks; two pooled HEK293T ATAC tracks,
                     GSE283384 and GSE152177). 78 TFs (ZNF768 has no genome-wide pre-mask matrix).
                     This is the mask used for the Fig 2 imputed panels.

Each <TF>/ folder: matrix_csr.npz (scipy sparse CSR, rows = bins, columns = cells, values 0/1),
barcodes.tsv (column order), meta.json (model and mask settings, call counts).
regions.tsv.gz: row order (chr:start-end), identical for every TF, stored once per mask folder.

Python:  import scipy.sparse as sp; m = sp.load_npz("ATAC_mask/CTCF/matrix_csr.npz")
