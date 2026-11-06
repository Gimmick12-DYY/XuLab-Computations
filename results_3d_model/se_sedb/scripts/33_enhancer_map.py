"""Number the enhancer-class partners of the SE hub in genomic order (the clockwise order of the chord diagram). Writes enhancer_map.tsv to both figure folders."""
from pathlib import Path
import pandas as pd
R0 = Path(__file__).resolve().parents[1]; T = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); T["pos"] = [sum(map(int, r.split(":")[1].split("-"))) // 2 for r in T.partner]; E = T[T.cls == "enhancer"].sort_values("pos").reset_index(drop=True); E = pd.DataFrame(dict(number=E.index + 1, name=[f"enhancer {i + 1}" for i in E.index], coordinates=E.partner, old_label=E.label, bin_10kb=E.pos // 10000, on_hi_c_loop_to_se=E.on_loop))
for d in ("figure_rosette_enh", "figure_chord_enh"): E.to_csv(R0 / d / "enhancer_map.tsv", sep="\t", index=False)
print(E.to_string(index=False))
