import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
D="/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ["Cambria.ttc","Cambriab.ttf","Cambriai.ttf"]: fm.fontManager.addfont(D+f)
plt.rcParams.update({"font.family":"Cambria","font.size":12})
d=json.load(open("figdata.json"))["noq"]
tf=np.array(d["tf"]); cells=np.array(d["cells"]); order=np.argsort(-cells); tf=tf[order]; cells=cells[order]; n=len(tf)
fig,ax=plt.subplots(figsize=(6.2,3.0),dpi=300)
x=np.arange(n)
ax.scatter(x,np.log10(cells),s=10,color="#BFBDB6",lw=0,zorder=2)
rng=np.random.default_rng(3)
for t,col in [("wiz","#8C1C13"),("znf644","#8C1C13")]:
    k=list(tf).index(t)
    lo,hi=max(0,k-4),min(n-1,k+4)
    ax.axvspan(lo-0.4,hi+0.4,color="#8C1C13",alpha=0.10,lw=0,zorder=1)
    ax.scatter([k],[np.log10(cells[k])],s=34,color=col,zorder=3)
    ax.annotate(t.upper(),(k,np.log10(cells[k])),xytext=(0,14),textcoords="offset points",ha="center",fontsize=10,color=col)
    j=rng.choice([i for i in range(lo,hi+1) if i!=k])
    ax.scatter([j],[np.log10(cells[j])],s=34,facecolor="white",edgecolor="#1E5E3B",lw=1.4,zorder=4)
ax.set_xlabel("TFs ranked by cell count (deepest → shallowest)"); ax.set_ylabel("log$_{10}$ cells")
ax.set_xticks([]); ax.spines[["top","right"]].set_visible(False)
ax.text(0.99,0.95,"red = group member;  shaded = ±4-rank pool\ngreen circle = one random replacement",transform=ax.transAxes,ha="right",va="top",fontsize=9.5,color="#595959")
fig.savefig("fig_rank_null.png",bbox_inches="tight",facecolor="white")
