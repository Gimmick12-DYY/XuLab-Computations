Raw single-cell TF binding data, HEK293T, hg38, 79 TFs.

fragments/  TF.<TF>.bed.gz: one line per read; column 4 = cell barcode.
            Same files as /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment.
bin/        <TF>_bin1000_mtx.rds: R sparse matrix of read counts, 1 kb genome bins (3,031,053) x cells of that TF.
            Built from the fragments; same bins and order as imputed/*/regions.tsv.gz.
peak/       TF.<TF>_peaks.q005.narrowPeak: MACS2 peaks called on the pooled reads of each TF (q < 0.05).
            Same files as /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks. Full MACS2 output (pileups, summits)
            is in TAGATG-293T/bigwig/macs2/.
