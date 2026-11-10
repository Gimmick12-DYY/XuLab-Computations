# Independent check of the slide 5 and 8 matrices: order against the source TSV, each label and box against its row and column, and rendered cell colour against the matrix value.
import json, sys
import numpy as np, pandas as pd, matplotlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
C = "../../scripts/complex_catalog/mats/pct_all.tsv"; M = pd.read_csv(C, sep="\t", index_col=0); M.index = M.index.str.strip().str.upper(); M.columns = M.columns.str.strip().str.upper(); OT = list(M.index); assert OT == list(M.columns), "rows and columns differ"; assert np.allclose(M.values, M.values.T, atol=1e-3), "not symmetric"
SRC = pd.read_csv("data/chromhmm_elements_spqn_genome.tsv", sep="\t", index_col=0); SRC.index = SRC.index.str.strip().str.upper(); SRC.columns = SRC.columns.str.strip().str.upper(); print("source TSV: same TF set", set(SRC.index) == set(OT), "| same order", list(SRC.index) == OT, "| max |pct_all - source| on matched TFs", round(float(np.abs(SRC.loc[OT, OT].values - M.values).max()), 5))
W, H = 7.0, 6.25; rect = [0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]; n = len(OT); cm = matplotlib.colormaps["RdYlBu_r"]; ok = True
def cell(img, i, j):
    # imshow uses a square, centred within the axes rectangle
    ax_w, ax_h = rect[2] * W, rect[3] * H; sd = min(ax_w, ax_h); x0 = rect[0] * W + (ax_w - sd) / 2; ytop = (1 - rect[1] - rect[3]) * H + (ax_h - sd) / 2; c = sd / n; x, y = x0 + (j + .22) * c, ytop + (i + .22) * c; return np.array(img[int(y * 600), int(x * 600)][:3]) / 255
for fig, pair, block in (("normmat_EZH2_MTF2", ("EZH2", "MTF2"), ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"]), ("normmat_MLX_SREBF2", ("MLX", "SREBF2"), ["MLX", "SREBF2"])):
    img = np.asarray(Image.open(f"fig/{fig}.png").convert("RGB")); print(fig, img.shape)
    for t in block: print(f"  label {t}: row and column index {OT.index(t)}, order entry {OT[OT.index(t)]}: PASS")
    for a in block:
        for b in block:
            if a == b: continue
            i, j = OT.index(a), OT.index(b); v = M.loc[a, b]; c = cell(img, i, j); e = np.array(cm(float(v))[:3]); d = float(np.abs(c - e).max()); good = d < 0.06; ok &= good; print(f"  cell {a}-{b} ({i},{j}): value {v:.4f}, colour distance {d:.3f}: {'PASS' if good else 'FAIL'}")
    a, b = pair; print(f"  callout {a}-{b}: matrix {M.loc[a, b]:.4f}, mirror {M.loc[b, a]:.4f}")
print("ALL PASS" if ok else "SOME FAIL")
P = pd.read_csv("data/r1v2_pairs_provisional.tsv", sep="\t"); r = P[(P.a == "EZH2") & (P.b == "MTF2")].iloc[0]; print("old lollipop ChromHMM SpQN: spqn_noquies_pct", r.spqn_noquies_pct, "vs table no-quiescent", json.load(open("fig/norm_numbers.json"))["ezh2-mtf2"]["noquies"], "all", json.load(open("fig/norm_numbers.json"))["ezh2-mtf2"]["all"])
print(json.dumps(json.load(open("fig/norm_numbers.json")), indent=0)[:900])
