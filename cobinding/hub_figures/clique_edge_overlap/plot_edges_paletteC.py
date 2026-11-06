"""Fig 3a (palette C): clique regions and clique edges by number of TFs whose cliques contain them."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
E = Path(__file__).resolve().parent
OUT = ROOT / "figures_paletteC"; OUT.mkdir(exist_ok=True)
k = pd.read_csv(E / "edges_and_regions_by_n_tfs.tsv", sep="\t")
MM = 1 / 25.4

fig, ax = plt.subplots(1, 2, figsize=(120 * MM, 45 * MM), gridspec_kw={"wspace": 0.3})
ax[0].bar(k.n_tfs, k.regions, color=P.neutral.light); ax[0].set_yscale("log"); ax[0].set_title("Clique regions", loc="left")
ax[1].bar(k.n_tfs, k.edges, color=P.neutral.dark); ax[1].set_yscale("log"); ax[1].set_title("Clique edges (same two regions)", loc="left")
for a in ax:
    a.set_xlabel("Number of TFs whose cliques contain it")
ax[0].set_ylabel("count (log)")
ne = int(k.edges.sum()); sh = int(k.loc[k.n_tfs > 1, "edges"].sum())
nr = int(k.regions.sum()); shr = int(k.loc[k.n_tfs > 1, "regions"].sum())
ax[0].text(0.97, 0.95, f"{shr:,} of {nr:,} regions ({100*shr/nr:.1f}%)\nin ≥2 TFs' cliques", transform=ax[0].transAxes, ha="right", va="top", fontsize=6)
ax[1].text(0.97, 0.95, f"{sh:,} of {ne:,} edges ({100*sh/ne:.1f}%)\nin ≥2 TFs' cliques", transform=ax[1].transAxes, ha="right", va="top", fontsize=6)
fig.suptitle("How often TFs' cliques reuse the same regions and the same edges (all cliques, blacklist removed)", x=0.01, y=1.06, ha="left", fontweight="bold")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"fig_edges_regions_by_n_tfs_paletteC.{ext}", bbox_inches="tight")
plt.close(fig)
print("ok", shr, nr, sh, ne)
