#!/usr/bin/env python
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _cfg import base_parser, load_config, resolve_paths

log=logging.getLogger("06_viz_fits"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    ap=base_parser("")
    ap.add_argument("--imputed-file", type=str, default=None)
    ap.add_argument("--n-regions", type=int, default=1000)
    ap.add_argument("--n-cells", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", type=str, default="fits_visualize")
    args=ap.parse_args()
    cfg=resolve_paths(load_config(args.config), args.work_dir)
    inp=Path(cfg["paths"]["fits_input"])/"fits_input.csv"
    if args.imputed_file: imp=Path(args.imputed_file)
    else: imp=Path(Path(cfg["paths"]["fits_run"])/"fits_imputed_path.txt").read_text().strip()
    x=np.loadtxt(inp, delimiter=",", dtype=np.float64)
    y=np.loadtxt(imp, delimiter=",", dtype=np.float64)
    if x.shape!=y.shape: raise SystemExit(f"shape mismatch {x.shape} vs {y.shape}")
    rng=np.random.default_rng(args.seed)
    score=(x>0).sum(axis=1)
    rp=np.argsort(score)[::-1][:min(args.n_regions,x.shape[0])]
    cp=rng.choice(x.shape[1], size=min(args.n_cells,x.shape[1]), replace=False)
    xs=(x[rp][:,cp]>0).astype(float)
    ys=y[rp][:,cp]
    vmax=float(np.quantile(ys,0.99)) if ys.size else 1.0
    if vmax<=0: vmax=float(ys.max() if ys.size else 1.0)
    out=Path(cfg["paths"]["eval"]); out.mkdir(parents=True,exist_ok=True)
    png=out/f"{args.out_name}.png"
    fig,ax=plt.subplots(1,2,figsize=(14,8),sharey=True)
    ax[0].imshow(xs,aspect="auto",cmap="Reds",vmin=0,vmax=1,interpolation="nearest")
    ax[0].set_title(f"Original binary ({xs.shape[0]}x{xs.shape[1]})")
    ax[1].imshow(ys,aspect="auto",cmap="Reds",vmin=0,vmax=vmax,interpolation="nearest")
    ax[1].set_title(f"FITS imputed (vmax=q99={vmax:.3g})")
    for a in ax: a.set_xlabel("cells")
    ax[0].set_ylabel("regions")
    fig.tight_layout(); fig.savefig(png,dpi=150); plt.close(fig)
    meta={"shape":[int(x.shape[0]),int(x.shape[1])],"orig_density":float((x>0).mean()),"imputed_mean":float(y.mean()),"imputed_vmax_q99":vmax}
    (out/f"{args.out_name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", png)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
