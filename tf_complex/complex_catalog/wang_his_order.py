# Our ChromHMM-18 (Dr. Wang h5ad) matrices drawn in the TF order of his own figures, on his colour scales,
# so the two can be compared cell by cell. Orders transcribed from his Pearson and Spearman slides.
import json, numpy as np
exec(open("matfig.py").read().split('if __name__=="__main__":')[0])
J = json.load(open("sp_mats.json")); T = J["tfs"]; W = J["mats"]["wang18"]
HIS = {"pearson": ("MEF2A THAP6 ZBTB40 FOXK1 POGK TRAFD1 ZNF143 ETS2 E4F1 ZNF746 HOXC9 SREBF2 DR1 ZUFSP HOXC4 MEF2D WIZ ZBTB43 AKAP8L MTF2 MLX SMARCA1 CHAMP1 MAZ EZH2 ZNF644 HOXB8 ZNF512 HP1 NRF1 HOXB6 ZBTB7A EPAS1 ADNP ZNF512B DNMT1 SOX4 ZIC2 FLYWCH1 SRCAP ZNF282 HOXA10 RBPJ FOXJ3 HOXD10 GLYR1 MBD3 ZNF777 HOXB9 PCGF6 ZNF606 RELA HES1 XPA CTCF STAG2 NEUROD4 E2F3 SAFB ZNF581 ZNF768 ZNF367 SOX12 NR1D2 SETDB1 MSX2 ZNF79 TBX3 PA2G4 DDIT3 RBBP4 RBBP7 ZNF503 ZNF703 ZNF507 FOXC1 NME2 HOXA5 NFYA", 0.65),
       "spearman": ("HOXC9 THAP6 HOXC4 ZNF143 ZNF746 ZBTB40 AKAP8L TRAFD1 MEF2D FOXK1 ETS2 POGK DR1 ZUFSP E4F1 MEF2A HOXA5 RBBP7 ZNF703 FOXC1 PA2G4 ZNF79 CTCF HES1 RBBP4 ZNF768 DDIT3 XPA NFYA NME2 TBX3 ZNF507 NRF1 ADNP SOX4 ZBTB7A SMARCA1 ZNF512B FLYWCH1 HOXB9 DNMT1 ZNF282 ZIC2 GLYR1 HOXD10 RBPJ SRCAP HOXA10 MBD3 ZNF777 HOXB8 EPAS1 ZBTB43 PCGF6 CHAMP1 ZNF644 HOXB6 MLX SREBF2 MAZ MTF2 WIZ EZH2 ZNF512 FOXJ3 HP1 ZNF606 MSX2 SAFB NR1D2 ZNF581 NEUROD4 ZNF367 ZNF503 RELA SOX12 STAG2 E2F3 SETDB1", 0.83)}
for metric, (order, vmin) in HIS.items():
    Lo = order.lower().split(); assert sorted(Lo) == sorted(T), set(T) ^ set(Lo)
    o = [T.index(t) for t in Lo]; R = np.array(W[metric])[np.ix_(o, o)]; n = len(Lo)
    fig = plt.figure(figsize=(6.4, 6.0), dpi=300); ax = fig.add_axes([0.13, 0.06, 0.74, 0.80])
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=vmin, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([t.upper() for t in Lo], rotation=90, fontsize=4.6); ax.set_yticklabels([t.upper() for t in Lo], fontsize=4.6)
    ax.tick_params(length=1.2, width=0.3, pad=1)
    ax.set_title(f"Ours, in Dr. Wang's {metric.capitalize()} order", fontsize=9)
    cax = fig.add_axes([0.905, 0.30, 0.018, 0.36]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6)
    cb.set_label("Spearman ρ" if metric == "spearman" else "Pearson r", fontsize=7)
    fig.savefig(f"wang_hisorder_{metric}.png", facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    # how block-like is his order on our data: mean r within his diagonal blocks vs between
    print(metric, "done")
