#!/usr/bin/env python
"""Continuous (pre-binarization, pre-mask) imputer scores, pseudobulk per bin, for the bins of r9_prep_bins.py.
Same model forward pass as unified/scripts/02_impute.py (yhat = gate * p), no threshold applied.
Per bin: mean yhat over cells and the fraction of cells with yhat > the TF's impute threshold (meta.json).
Panel TFs use all cells; other TFs a random 3000 (seed 7).
Usage: r9a_continuous_scores.py <TF> [max_cells]. Output unified/work/<tf>/predict/pseudobulk_audit_bins.npz."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/work/users/d/y/dyy12/XuLab")
sys.path.insert(0, str(ROOT / "unified/scripts"))
from _cfg import load_config  # noqa: E402
from _model import GatedUnifiedModel  # noqa: E402

tf = sys.argv[1].lower()
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
w = ROOT / "unified/work" / tf
cfgp = ROOT / f"unified/configs/{tf}.yaml"
if not cfgp.exists():
    cfgp = ROOT / "unified/configs/default.yaml"
cfg = load_config(cfgp)
m = cfg["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_cells = sum(1 for _ in open(w / "seqs/barcodes.tsv"))
meta = json.loads((w / "impute/meta.json").read_text())
thr = float(meta["threshold"])
ck = torch.load(w / "model/ckpt.pt", map_location=device)
ckm = (ck.get("config") or {}).get("model") or {}
sa = bool(ckm.get("sparsity_aware", m.get("sparsity_aware", False)))
assert not sa, "sparsity_aware ckpt not handled"
model = GatedUnifiedModel(
    n_cells=n_cells, seq_length=int(cfg["seqs"].get("seq_length", 768)), latent_dim=int(m.get("latent_dim", 64)),
    stem_filters=int(m.get("stem_filters", 288)), stem_kernel=int(m.get("stem_kernel", 17)),
    stem_pool=int(m.get("stem_pool", 3)), tower_channels=tuple(m.get("tower_channels", [288, 323, 363, 407])),
    tower_kernel=int(m.get("tower_kernel", 5)), head_filters=int(m.get("head_filters", 256)),
    dropout=float(m.get("dropout", 0.20)), context_dilations=tuple(m.get("context_dilations", [1, 4, 16])),
    context_kernel=int(m.get("context_kernel", 3)), alpha_init=float(m.get("alpha_init", 0.1)),
    gate_hidden=tuple(m.get("gate_hidden", [128, 64, 32])), gate_use_concat=bool(m.get("gate_use_concat", True)),
    sparsity_aware=False, use_cell_depth=bool(ckm.get("use_cell_depth", m.get("use_cell_depth", True))),
    use_obs_count=bool(ckm.get("use_obs_count", m.get("use_obs_count", True)))).to(device)
model.load_state_dict(ck["state_dict"])
model.eval()
OC = ROOT / "results_293T_analysis/fig2_qc/imputed_audit/continuous"
want = np.load(OC / "bins_mm.npy")
h5 = h5py.File(w / "seqs/seqs.h5", "r")
mm = np.asarray(h5["mm_row_idx"])
pos = np.full(int(mm.max()) + 1, -1, np.int64)
pos[mm] = np.arange(len(mm))
want_ok = want[want <= mm.max()]
loc = pos[want_ok]
keep = loc >= 0
order = np.argsort(loc[keep])
h5_idx = loc[keep][order]
out_row = np.flatnonzero(want <= mm.max())[keep][order]
cells = None
if cap and n_cells > cap:
    cells = torch.from_numpy(np.sort(np.random.default_rng(7).choice(n_cells, cap, replace=False))).to(device)
meanv = np.full(len(want), np.nan, np.float32)
fracv = np.full(len(want), np.nan, np.float32)
CHUNK = 512
with torch.no_grad():
    for i in range(0, len(h5_idx), CHUNK):
        ii = h5_idx[i:i + CHUNK]
        X = torch.from_numpy(h5["X"][ii].astype(np.float32)).to(device)
        e = model.encode_bins(X)
        y = model.predict_chunk(e, cell_idx=cells)
        meanv[out_row[i:i + CHUNK]] = y.mean(1).cpu().numpy()
        fracv[out_row[i:i + CHUNK]] = (y > thr).float().mean(1).cpu().numpy()
        if i % (CHUNK * 40) == 0:
            print(tf, i, len(h5_idx), flush=True)
(w / "predict").mkdir(exist_ok=True)
np.savez(w / "predict/pseudobulk_audit_bins.npz", bins_mm=want, mean_score=meanv, frac_above_thr=fracv,
         threshold=thr, n_cells_used=(cap if cells is not None else n_cells), n_cells_total=n_cells)
print("done", tf, n_cells, "nan bins", int(np.isnan(meanv).sum()))
