"""DRAFT panel j for review (user 2026-11-18): GREAT-style region enrichment (scripts/go_great_regions.py, checks/go_great_regions,
commit 3ac8dff) BEFORE the random-peak and open-chromatin controls. Left: TF-specific module elements (modules of 1-2 TFs) per TF, rows =
TFs with >= 1 term q < 0.05 ordered by best q, each TF's top 2 terms; x = -log10 q, size = foreground regions. Right: conserved elements
(>= 3 TFs), per-TF summary: the 12 terms significant in the most TFs; x = number of TFs with q < 0.05, size = median regions.
  python Figure3/scripts/j_great_draft.py   (Mac)"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_great_regions"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
TEAL = "#2C7A6B"
S = pd.read_csv(G / "per_TF_q005.tsv", sep="\t"); S = S[~S.cluster_artifact.fillna(False).astype(bool)]
def short(s, n=46):
    s = s.replace("positive regulation of", "pos. reg. of").replace("negative regulation of", "neg. reg. of").replace("regulation of", "reg. of")
    s = s.replace("biosynthetic process", "biosynthesis").replace("metabolic process", "metabolism").replace(" signaling pathway", " signaling")
    s = s[0].upper() + s[1:]; return s if len(s) <= n else s[:n - 1] + "…"
sp = S[S.group == "specific"].sort_values("q"); order = sp.groupby("tf").q.min().sort_values().index
L = pd.concat([sp[sp.tf == t].head(2) for t in order]).reset_index(drop=True)
cs = S[S.group == "conserved"]
R = cs.groupby(["term", "name"]).agg(n_tfs=("tf", "nunique"), med_regions=("fg_regions", "median"), best_q=("q", "min")).reset_index().sort_values(["n_tfs", "best_q"], ascending=[False, True]).head(12)
n_sp_tested = 50; n_cs_tested = 41
fig = plt.figure(figsize=(183 * MM, (18 + 2.75 * len(L)) * MM)); hL = len(L); hR = len(R)
a = fig.add_axes([0.33, 0.06, 0.17, 0.88]); b = fig.add_axes([0.80, 0.06 + 0.88 * (1 - hR / hL), 0.15, 0.88 * hR / hL])
sz = lambda n: 3 + 0.08 * np.asarray(n)
y = np.arange(hL); a.scatter(-np.log10(L.q), y, s=sz(L.fg_regions), color=TEAL, lw=0, zorder=3)
a.set_yticks(y); a.set_yticklabels([f"{t}  ·  {short(n)}" if (i == 0 or L.tf[i - 1] != t) else short(n) for i, (t, n) in enumerate(zip(L.tf, L.name))], fontsize=5.5)
a.set_ylim(hL - 0.5, -0.7); a.tick_params(axis="y", length=0); a.axvline(-np.log10(0.05), color="#9E9E9E", lw=0.6, ls=(0, (3, 2)))
for i in range(1, hL):
    if L.tf[i] != L.tf[i - 1]: a.axhline(i - 0.5, color="#EEEEEE", lw=0.5, zorder=0)
a.set_xlabel("−log10 q"); a.set_title(f"TF-specific module elements (1–2 TFs), per TF\n{L.tf.nunique()} of {n_sp_tested} TFs with ≥1 term", fontsize=6.5, pad=4)
yr = np.arange(hR); b.scatter(R.n_tfs, yr, s=sz(R.med_regions), color=TEAL, lw=0, zorder=3)
b.set_yticks(yr); b.set_yticklabels([short(n) for n in R.name], fontsize=5.5); b.set_ylim(hR - 0.5, -0.7); b.tick_params(axis="y", length=0)
b.set_xlim(0, R.n_tfs.max() + 2); b.set_xlabel("TFs with q < 0.05"); b.xaxis.grid(True, color="#EEEEEE", lw=0.5); b.set_axisbelow(True)
b.set_title(f"Conserved module elements (≥3 TFs)\n{cs.tf.nunique()} of {n_cs_tested} TFs with ≥1 term", fontsize=6.5, pad=4)
hs = [plt.scatter([], [], s=sz(n), color=TEAL, lw=0) for n in (10, 50, 200)]
b.legend(hs, ["10", "50", "200 regions"], loc="upper left", bbox_to_anchor=(-0.9, -0.12 * hL / hR), ncol=3, fontsize=5.5, handletextpad=0.2, columnspacing=0.8)
fig.text(0.99, 0.995, "draft, before random-peak and open-chromatin controls", ha="right", va="top", fontsize=5, color="#9E9E9E")
PS = "ArialMT"
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size = [], None
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font": mm_ = re.match(r"(?:\S+\s+)?([\d.]+)px", v); size = mm_.group(1) if mm_ else None
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS}, Helvetica, sans-serif" font-size="{float(size):.3f}">'
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = H / "j_great_draft"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("rows", hL, hR, "font left:", p.read_text().count("font: "))
