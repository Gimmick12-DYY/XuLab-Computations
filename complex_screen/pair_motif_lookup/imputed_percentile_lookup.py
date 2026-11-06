"""Shared-motif lookup on imputed result sets using cross-TF percentiles.

For each result set, build a TF x motif matrix of -log10 p (HOMER knownResults
Log P-value is natural log; AME uses adj_p).  For each motif, rank the 79 TFs
(absent TF = p 1).  A motif is 'shared-enriched' for pair A-B when both TFs are
at or above the given cross-TF percentile for that motif.  This replaces the
panel-frequency filter, which rejects every motif that imputed (ATAC-like)
peaks carry in most TFs, including CTCF's own motif.
"""
import csv, glob, math, os, sys, collections

# Usage: python imputed_percentile_lookup.py [MOTIF_ANALYSIS_DIR] [OUT_DIR] [PCT]
# Defaults run in place on Longleaf: reads XuLab/motif_analysis/{motif,motif_bins}/<tf>/...,
# tf_cells.tsv and pair_verdict.tsv next to this script, writes next to this script.
# Local mirror: python imputed_percentile_lookup.py inputs outputs/imputed_percentile
HERE = os.path.dirname(os.path.abspath(__file__))
inp = sys.argv[1] if len(sys.argv) > 1 else '/work/users/d/y/dyy12/XuLab/motif_analysis'
out = sys.argv[2] if len(sys.argv) > 2 else HERE
PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.90
os.makedirs(out, exist_ok=True)

tfs = sorted(os.path.basename(p) for p in glob.glob(f'{inp}/motif/*') if os.path.isdir(p))
cells = {}
cells_f = next((f for f in (f'{inp}/tf_cells.tsv', f'{HERE}/tf_cells.tsv') if os.path.exists(f)), None)
if cells_f:
    for r in csv.DictReader(open(cells_f), delimiter='\t'):
        k = [c for c in r if 'tf' in c.lower()][0]; v = [c for c in r if 'cell' in c.lower()][0]
        cells[r[k].lower()] = r[v]

def read_homer(path):
    d = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        lp = float(r['Log P-value'])
        d[r['Motif Name']] = -lp / math.log(10)
    return d

def read_ame(path):
    d = {}
    for line in open(path):
        if line.startswith('#') or not line.strip(): continue
        f = line.rstrip('\n').split('\t')
        if f[0] == 'rank': continue
        try: d[f[2]] = -math.log10(max(float(f[6]), 1e-300))
        except Exception: pass
    return d

sets = {
  'B_imp_peaks_homer': lambda t: (f'{inp}/motif/{t}/homer/knownResults.txt', read_homer),
  'B_imp_peaks_homer_ocr': lambda t: (f'{inp}/motif/{t}/homer_ocr/knownResults.txt', read_homer),
  'C_imp_bins_homer': lambda t: (f'{inp}/motif_bins/{t}/homer/knownResults.txt', read_homer),
  'B_imp_peaks_ame': lambda t: (f'{inp}/motif/{t}/ame/{t}/ame.tsv', read_ame),
  'B_imp_peaks_ame_ocr': lambda t: (f'{inp}/motif/{t}/ame_ocr/{t}/ame.tsv', read_ame),
  'C_imp_bins_ame': lambda t: (f'{inp}/motif_bins/{t}/ame/{t}/ame.tsv', read_ame),
}

pairs = list(csv.DictReader(open(next(f for f in (f'{HERE}/pair_verdict.tsv', 'outputs/pair_verdict.tsv') if os.path.exists(f))), delimiter='\t'))
summary = open(f'{out}/pair_shared_percentile_summary.tsv', 'w')
summary.write('pair\ta\tb\tset\taf3_tier\tpearson_domain_pct\tn_shared_pct%d\ttop_shared\tn_own_a\tn_own_b\n' % int(PCT*100))
rowsout = open(f'{out}/pair_shared_percentile_rows.tsv', 'w')
rowsout.write('pair\tset\tmotif\tnlogp_a\tnlogp_b\tpct_a\tpct_b\tn_tfs_above_pct\n')
bg = {}
for sname, fn in sets.items():
    mat = {}
    for t in tfs:
        path, reader = fn(t)
        if not os.path.exists(path): 
            alt = path.replace(f'/{t}/ame.tsv', '/ame.tsv')
            if os.path.exists(alt): path = alt
            else: continue
        try: mat[t] = reader(path)
        except Exception as e: print('skip', path, e, file=sys.stderr)
    present = sorted(mat)
    motifs = sorted(set(m for d in mat.values() for m in d))
    # cross-TF percentile per motif (among present TFs)
    pct = {}
    for m in motifs:
        vals = [(mat[t].get(m, 0.0), t) for t in present]
        order = sorted(vals)
        n = len(order)
        for i, (v, t) in enumerate(order):
            # percentile = fraction of TFs with strictly lower value
            lo = sum(1 for v2, _ in order if v2 < v)
            pct[(t, m)] = lo / max(n - 1, 1)
    n_above = collections.Counter(m for (t, m), p in pct.items() if p >= PCT)
    # per-TF count of motifs above PCT (depth check)
    with open(f'{out}/{sname}_tf_counts.tsv', 'w') as fh:
        fh.write('tf\tcells\tn_motifs_above_pct\n')
        for t in present:
            fh.write(f"{t}\t{cells.get(t,'')}\t{sum(1 for m in motifs if pct[(t,m)]>=PCT)}\n")
    nshared_all = []
    for r in pairs:
        a, b = r['a'].lower(), r['b'].lower()
        if a == 'hp1': a = 'hp1'
        if a not in mat or b not in mat: continue
        shared = []
        for m in motifs:
            pa, pb = pct.get((a, m), 0), pct.get((b, m), 0)
            if pa >= PCT and pb >= PCT and mat[a].get(m, 0) >= 5 and mat[b].get(m, 0) >= 5:
                shared.append((min(mat[a][m], mat[b][m]), m, pa, pb))
                rowsout.write(f"{r['pair']}\t{sname}\t{m}\t{mat[a][m]:.1f}\t{mat[b][m]:.1f}\t{pa:.2f}\t{pb:.2f}\t{n_above[m]}\n")
        shared.sort(reverse=True)
        own = lambda t, m: t.upper().replace('HP1','CBX') in m.upper()
        n_own_a = sum(1 for _, m, _, _ in shared if own(a, m))
        n_own_b = sum(1 for _, m, _, _ in shared if own(b, m))
        top = ';'.join(f"{m.split('/')[0]}({pa:.2f},{pb:.2f})" for _, m, pa, pb in shared[:4])
        summary.write(f"{r['pair']}\t{r['a']}\t{r['b']}\t{sname}\t{r['af3_tier']}\t{r['pearson_domain_pct']}\t{len(shared)}\t{top}\t{n_own_a}\t{n_own_b}\n")
        nshared_all.append((len(shared), r['af3_tier'], r['pair']))
    tot = len(nshared_all)
    k = sum(1 for n, _, _ in nshared_all if n > 0)
    af = [(n, p) for n, t, p in nshared_all if t in ('strong', 'weak')]
    print(f"{sname}: TFs {len(present)}, motifs {len(motifs)}, pairs {tot}, with >=1 shared@{PCT}: {k} ({100*k/max(tot,1):.1f}%), AF3-kept {sum(1 for n,_ in af if n>0)}/{len(af)}")
summary.close(); rowsout.close()
