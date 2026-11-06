import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon, Rectangle
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
D="/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ["Cambria.ttc","Cambriab.ttf","Cambriai.ttf"]: fm.fontManager.addfont(D+f)
plt.rcParams.update({"font.family":"Cambria"})
exec(open("groups_def.py").read())
RED="#8C1C13"
def load(path):
    with open(path) as fh: hdr=fh.readline().rstrip("\n").split("\t")[1:]
    R=np.loadtxt(path,delimiter="\t",skiprows=1,usecols=range(1,len(hdr)+1))
    return hdr,R
def order_of(R):
    Dm=1-R; Dm=(Dm+Dm.T)/2; np.fill_diagonal(Dm,0); Dm[Dm<0]=0
    y=squareform(Dm,checks=False); Z=linkage(y,"average"); Z=optimal_leaf_ordering(Z,y); return leaves_list(Z)
def runs(pos):
    pos=sorted(pos); out=[[pos[0],pos[0]]]
    for q in pos[1:]:
        if q==out[-1][1]+1: out[-1][1]=q
        else: out.append([q,q])
    return out
def draw(L,R,o,members,out,vmin,vmax,cblabel,fs=5.0,size=6.4,cmap="RdBu_r",inset=False):
    Lo=[L[i] for i in o]; Ro=R[np.ix_(o,o)]; n=len(Lo)
    pos=[Lo.index(m) for m in members]
    fig=plt.figure(figsize=(size,size*0.93),dpi=300)
    ax=fig.add_axes([0.13,0.06,0.74,0.80])
    im=ax.imshow(Ro,cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest")
    ax.set_xlim(-0.5,n-0.5); ax.set_ylim(n-0.5,-0.5)
    if not inset:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([t.upper() for t in Lo],rotation=90,fontsize=fs); ax.set_yticklabels([t.upper() for t in Lo],fontsize=fs)
        for lab_list in (ax.get_xticklabels(),ax.get_yticklabels()):
            for k,lab in enumerate(lab_list):
                if k in pos: lab.set_color(RED); lab.set_fontweight("bold"); lab.set_fontsize(fs+1.0)
        ax.tick_params(length=1.2,width=0.3,pad=1)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    lw=1.6 if not inset else 1.2
    rr=runs(pos)
    for a,b in rr:
        if b>a:
            ax.add_patch(Polygon([(a-0.5,a-0.5),(b+0.5,a-0.5),(b+0.5,b+0.5)],closed=True,fill=False,edgecolor="black",lw=lw,zorder=5))
    if len(rr)>1 or all(a==b for a,b in rr):
        # member x member cells (off-diagonal) outlined, plus edge markers
        for i in pos:
            for j in pos:
                if i<j and not any(a<=i and j<=b for a,b in rr):
                    ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="black",lw=0.9,zorder=6))
    if True:
        for p_ in pos:
            ax.plot([p_],[-1.6],marker="v",ms=4.2 if not inset else 3,color=RED,clip_on=False,zorder=7)
            ax.plot([n+0.6],[p_],marker="<",ms=4.2 if not inset else 3,color=RED,clip_on=False,zorder=7)
    for s in ax.spines.values(): s.set_linewidth(0.4)
    cax=fig.add_axes([0.905,0.30,0.018,0.36]); cb=fig.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=6,width=0.3,length=2); cb.outline.set_linewidth(0.3)
    cb.set_label(cblabel,fontsize=7)
    fig.savefig(out,facecolor="white",bbox_inches="tight",pad_inches=0.04); plt.close(fig)
    return pos,rr
if __name__=="__main__":
    L,R=load("mats/enrich.tsv"); o=order_of(R)
    Ln,Rn=load("mats/noqS.tsv"); assert Ln==L
    on=order_of(Rn)
    Lo=[L[i] for i in o]
    print("order:"," ".join(Lo))
    import json; json.dump({"order":Lo,"order_noqS":[L[i] for i in on]},open("orders.json","w"))
    off=Rn[~np.eye(len(L),dtype=bool)]
    for key,name,mem in GROUPS:
        pos,rr=draw(L,R,o,mem,f"mat_{key}.png",-0.6,0.9,"Enrichment Pearson r")
        p2,rr2=draw(L,Rn,on,mem,f"ins_{key}.png",np.quantile(off,0.02),np.quantile(off,0.995),"SpQN r",size=3.0,inset=True)
        print(key,"runs enrich",rr,"| runs noqS",rr2)
