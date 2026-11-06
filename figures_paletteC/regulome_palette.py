"""Paper palette, option C (Continuity). Import and call use() before plotting: P = rp.use()."""
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

NEUTRAL = {"ink": "#1A1A1A", "dark": "#4D4D4D", "mid": "#8C8C8C", "light": "#BDBDBD", "pale": "#E3E3E3", "faint": "#F2F2F2"}
EVIDENCE = {"hic_loop": "#B83A2E", "cicero_only": "#7A9CCB"}   # shared edge on a 0.7 loop / Cicero only
ARMS = {"raw": "#D98C1F", "imputed": "#2C7A6B", "atac": "#6A51A3", "bulk": NEUTRAL["ink"]}
CHROMHMM = {"promoter": "#B03A2E", "active_enh": "#E8821E", "weak_enh": "#F6CB8F", "tss_flank": "#D46A5E",
            "transcribed": "#7FA6CE", "repressed": "#55606E", "quiescent": "#E3E3E3"}
HIGHLIGHT = "#FCE9A8"
SEQ = ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
DIV = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"]
LOOP_STRATA = ["#E3E3E3", "#FCBBA1", "#FC9272", "#EF3B2C", "#CB181D", "#67000D"]   # no loop edge, >0-25, 25-50, 50-75, 75-<100, 100%
NA = NEUTRAL["faint"]


def seq_cmap(name="regulome_seq"):
    c = LinearSegmentedColormap.from_list(name, SEQ); c.set_bad(NA); return c


def div_cmap(name="regulome_div"):
    c = LinearSegmentedColormap.from_list(name, DIV); c.set_bad(NA); return c


class _P(dict):
    __getattr__ = dict.__getitem__


def use():
    ink = NEUTRAL["ink"]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.titlesize": 7, "axes.labelsize": 7, "figure.titlesize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": ink, "axes.titlecolor": ink,
        "xtick.color": ink, "ytick.color": ink, "legend.frameon": False,
        "savefig.dpi": 600, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return _P(neutral=_P(NEUTRAL), evidence=_P(EVIDENCE), arms=_P(ARMS), chromhmm=_P(CHROMHMM), highlight=HIGHLIGHT,
              seq=SEQ, div=DIV, loop_strata=LOOP_STRATA, na=NA, seq_cmap=seq_cmap, div_cmap=div_cmap)
