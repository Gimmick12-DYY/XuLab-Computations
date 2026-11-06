"""Comparison table (Arial 7 pt, live SVG text) of the five chr16 models on common held-out sets (12f_compare.py)."""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
F = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst"); T = pd.read_csv(F / "compare/model_comparison.tsv", sep="\t")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
inputs = {"TF only (v2)": "TF summits", "loops + TF": "TF + 165 loops", "loops only": "165 loops", "Hi-C first": "Hi-C pixels + loops + red edges",
          "Hi-C first + TF": "Hi-C pixels + loops + red edges + TF"}
cols = ["model", "inputs", "O/E held-out\npixels (176)", "O/E unrestrained\npixels (13,509)", "blue hub edges\nvs dist / vs bound (5)",
        "held-out shared\nvs dist / vs bound (29)", "Cicero step-1\nvs dist / vs bound", "≥3 hub cores in one\nTF flower (null)"]
rows = []
for r in T.itertuples():
    fl = f"{r.hub3*100:.0f}% ({r.hub3_null*100:.0f}%)" if r.hub3 > 0 else "no TF clusters"
    rows.append([r.model, inputs[r.model], f"{r.oe_heldout:.2f}", f"{r.oe_unrestrained:.2f}", f"{r.hub_edges_dist:.1f}× / {r.hub_edges_bound:.2f}×",
                 f"{r.shared_edges_dist:.1f}× / {r.shared_edges_bound:.2f}×", f"{r.cicero_step1_dist:.1f}× / {r.cicero_step1_bound:.2f}×", fl])
fig = plt.figure(figsize=(250 / 25.4, 62 / 25.4)); ax = fig.add_axes([0, 0.02, 1, 0.96]); ax.set_axis_off()
tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colWidths=[0.1, 0.19, 0.09, 0.1, 0.13, 0.13, 0.13, 0.13])
tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 2.0)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor("#CCCCCC"); c.set_linewidth(0.4)
    if i == 0: c.set_facecolor("#F2F2F2"); c.set_text_props(weight="bold")
    if i == 5: c.set_facecolor("#FDEBD7")
for e in ("svg", "pdf"): fig.savefig(F / f"compare/model_table.{e}", transparent=True)
fig.savefig(F / "compare/model_table_preview.png", dpi=250, facecolor="white"); print("ok")
