import numpy as np, json, matplotlib
from matplotlib.lines import Line2D
exec(open("matfig.py").read().split('if __name__=="__main__":')[0])
exec(open("known_def.py").read())
S=json.load(open("bymatrix.json"))
COLS=["#000000","#6A1B9A","#00695C","#C2185B","#1565C0","#E65100","#2E7D32"]
SHORT={"prc2":"PRC2","nurd":"NuRD","nurd_z":"NuRD+Z512B","cohesin":"Cohesin","maz":"MAZ–ZNF143","g9a":"G9a/GLP","nfy":"NF-Y/MLX","mef2":"MEF2","znf512":"ZNF512","pcgf6":"PRC1.6","chahp":"ChAHP","hp1setdb1":"HP1/SETDB1","e2f3rbbp4":"E2F3–RBBP4","hox":"HOX"}
def overview(mk,out,fs=5.0,size=6.4):
    L,R=load(f"mats/pct_{mk}.tsv"); n=len(L)
    rec=[(k,name,mem) for k,name,mem in KNOWN if S[mk][k]["rec"]]
    fig=plt.figure(figsize=(size,size*0.93),dpi=300)
    ax=fig.add_axes([0.13,0.06,0.74,0.80])
    im=ax.imshow(R,cmap="RdYlBu_r",vmin=0,vmax=1,interpolation="nearest")
    ax.set_xlim(-0.5,n-0.5); ax.set_ylim(n-0.5,-0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([t.upper() for t in L],rotation=90,fontsize=fs); ax.set_yticklabels([t.upper() for t in L],fontsize=fs)
    allm=set(); handles=[]
    for c,(k,name,mem) in enumerate(rec):
        col=COLS[c%len(COLS)]; pos=sorted(L.index(t) for t in mem); allm|=set(pos)
        rr=runs(pos)
        for a,b in rr:
            if b>a: ax.add_patch(Polygon([(a-0.5,a-0.5),(b+0.5,a-0.5),(b+0.5,b+0.5)],closed=True,fill=False,edgecolor=col,lw=1.8,zorder=5))
        cells=[(i,j) for i in pos for j in pos if i<j and not any(a<=i and j<=b for a,b in rr)]
        d=0.22 if k=="nurd_z" else 0.0
        for i,j in cells: ax.add_patch(Rectangle((j-0.5+d,i-0.5+d),1-2*d,1-2*d,fill=False,edgecolor=col,lw=1.3,zorder=6 if d==0 else 7))
        handles.append(Line2D([0],[0],color=col,lw=2.2,label=SHORT[k]))
    for lab_list in (ax.get_xticklabels(),ax.get_yticklabels()):
        for k_,lab in enumerate(lab_list):
            if k_ in allm: lab.set_fontweight("bold"); lab.set_color("#8C1C13")
    ax.tick_params(length=1.2,width=0.3,pad=1)
    for s_ in ax.spines.values(): s_.set_linewidth(0.4)
    cax=fig.add_axes([0.905,0.30,0.018,0.36]); cb=fig.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=6,width=0.3,length=2); cb.outline.set_linewidth(0.3)
    cb.set_label("Percentile of TF–TF Pearson r",fontsize=7)
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(0.885,0.05),fontsize=7,frameon=False,handlelength=1.5,title="Recovered",title_fontsize=7.5)
    fig.savefig(out,facecolor="white",bbox_inches="tight",pad_inches=0.04); plt.close(fig)
for mk,_,_ in MATS:
    overview(mk,f"ov_{mk}.png")
    L,R=load(f"mats/pct_{mk}.tsv"); o=np.arange(len(L))
    for k,name,mem in KNOWN:
        if S[mk][k]["rec"]:
            draw(L,R,o,mem,f"bm_{mk}_{k}.png",0,1,"Percentile of TF–TF Pearson r",cmap="RdYlBu_r")
print("done")
