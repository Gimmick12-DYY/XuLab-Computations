#!/usr/bin/env python
"""Select the 1 kb bins needed for the continuous-score audit: summit bin +-2 for every summit used by the 6x6
(round-2 summits, preferred 2x; top 5000 all chromosomes, top 2500 per chromosome half) and the HEK293T 2x2 (top 5000).
Writes OUT/imputed_audit/continuous/: bins_frame.npy (frame index), bins_mm.npy (unified regions/mm row index),
tf_list.txt (6 panel TFs + 20 random others with a checkpoint, seed 2026; ZNF768 excluded, no unmasked matrix)."""
import numpy as np
import pandas as pd

from fig2_common import OUT, ROOT, TFS, bin_index, INP

OC = OUT / "imputed_audit" / "continuous"
OC.mkdir(parents=True, exist_ok=True)
bi, off, N = bin_index()
reg = pd.read_csv(ROOT / "unified/work/ctcf/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
fb = np.array([off[c][0] for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
inv = np.full(N, -1, np.int64)
inv[fb] = np.arange(len(fb))


def pick(S, mask, tfs, spec):
    cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
    h = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
    out = []
    for half, n in spec:
        for t in tfs:
            ix = np.flatnonzero((S.ref.values == t) & mask & h[half])
            out.append(ix[np.argsort(-S.signal.values[ix], kind="stable")][:n])
    return np.unique(np.concatenate(out))


S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
S2 = pd.read_csv(OUT / "hek293t_2x2/summits.tsv", sep="\t", low_memory=False)
b = []
for S, ix in ((S6, pick(S6, S6.bulk_spec.values > 1, TFS, (("all", 5000), ("odd", 2500), ("even_X", 2500)))),
              (S2, pick(S2, S2.preferred.values.astype(bool), ["CTCF", "ZNF282"], (("all", 5000), ("odd", 5000), ("even_X", 5000))))):
    c = np.array([off[x][0] for x in S.chrom.values[ix]]) + S.pos.values[ix] // 1000
    b.append(c)
b = np.concatenate(b)
all_bins = np.unique(np.concatenate([b + d for d in (-2, -1, 0, 1, 2)]))
all_bins = all_bins[(all_bins >= 0) & (all_bins < N)]
all_bins = all_bins[inv[all_bins] >= 0]
np.save(OC / "bins_frame.npy", all_bins)
np.save(OC / "bins_mm.npy", inv[all_bins])
tf79 = [str(t) for t in np.load(INP / "frame.npz", allow_pickle=True)["tf79"]]
others = sorted(t for t in tf79 if t not in TFS and t != "ZNF768" and (ROOT / f"unified/work/{t.lower()}/model/ckpt.pt").exists())
rng = np.random.default_rng(2026)
extra = sorted(rng.choice(others, 20, replace=False).tolist())
(OC / "tf_list.txt").write_text("\n".join(list(TFS) + extra) + "\n")
print("bins", len(all_bins), "extra TFs", extra)
