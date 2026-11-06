import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
D="/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ["Cambria.ttc","Cambriab.ttf","Cambriai.ttf"]: fm.fontManager.addfont(D+f)
plt.rcParams.update({"font.family":"Cambria","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
d=json.load(open("figdata.json"))
fig=plt.figure(figsize=(8.6,4.6),dpi=300)
gs=fig.add_gridspec(2,4,width_ratios=[1.25,1.25,1,1],wspace=0.7,hspace=0.75)
for k,(tag,ttl) in enumerate([("noq","Raw Pearson"),("noqS","SpQN")]):
    ax=fig.add_subplot(gs[:,k])
    x=np.log10(d[tag]["cells"]); y=np.array(d[tag]["meanr"])
    ax.scatter(x,y,s=12,color="#5F5E5A",alpha=0.8,lw=0)
    rho=np.corrcoef(x,y)[0,1]
    for t,xx,yy in zip(d[tag]["tf"],x,y):
        off={"rbbp4":(-8,-22),"nfya":(-34,8),"ctcf":(14,-14),"wiz":(14,-16)}
        if t in off: ax.annotate(t.upper(),(xx,yy),fontsize=7,xytext=off[t],textcoords="offset points",arrowprops=dict(arrowstyle="-",color="#8C1C13",lw=0.6)); ax.scatter([xx],[yy],s=14,color="#8C1C13",lw=0)
    ax.set_title(f"{chr(65+k)}  {ttl}\nr(log cells, mean r) = {rho:.2f}",fontsize=11,loc="left")
    ax.set_xlabel("log$_{10}$ cell count"); ax.set_ylabel("TF mean r to other TFs")
names=["WIZ/ZNF644","NFYA/MLX/SREBF2","NuRD core","CTCF/STAG2"]
for j,nm in enumerate(names):
    ax=fig.add_subplot(gs[j//2,2+j%2])
    v=d["noq"]["nulls"][nm]; nl=np.array(v["null"])
    ax.hist(nl,bins=40,color="#BFBDB6",lw=0)
    ax.axvline(v["obs"],color="#8C1C13",lw=1.5)
    ax.set_title(nm,fontsize=9.5,loc="left"); ax.tick_params(labelsize=7.5); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if j>=2: ax.set_xlabel("mean r (group)",fontsize=8.5)
fig.text(0.585,0.965,"C  Depth-matched null, no-quiescent raw (5,000 draws; red = observed)",fontsize=11)
fig.savefig("fig_depth_null.png",bbox_inches="tight",facecolor="white")
