#!/usr/bin/env python
"""Re-threshold the unified imputer from its per-cell scores (no retraining), for the audit bins (r9_prep_bins.py).
Rules declared before results:
  a  one common cutoff for every TF = median of the 6 panel TFs' current cutoffs (impute/meta.json "threshold")
  b  per-TF cutoff giving every TF the same fraction of positive cell-bins = median over the 6 panel TFs of
     predicted_nnz_pre_mask / (n_bins x n_cells); cutoff = that upper quantile of yhat over a genome-wide sample of
     kept bins (same sampling as 02_impute.py pass 1, ~4M values, seed 0)
Per bin: positive cells = (yhat > cutoff) OR raw > 0 (keep_raw), as fractions of the cells used; masked version keeps
only raw positives outside the ATAC union. Panel TFs: all cells; other TFs: the same 3000 cells as r9a (seed 7).
Usage: r10a_rethreshold.py <TF> [max_cells]. Output unified/work/<tf>/predict/rethreshold_audit_bins.npz."""
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

ROOT = Path("/work/users/d/y/dyy12/XuLab")
sys.path.insert(0, str(ROOT / "unified/scripts"))
from _cfg import load_config  # noqa: E402
from _model import GatedUnifiedModel  # noqa: E402

PANEL = ["ctcf", "maz", "zbtb7a", "zic2", "znf282", "znf777"]
tf = sys.argv[1].lower()
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
w = ROOT / "unified/work" / tf
pm = [json.loads((ROOT / f"unified/work/{p}/impute/meta.json").read_text()) for p in PANEL]
CUT_A = float(np.median([m["threshold"] for m in pm]))
FRAC_B = float(np.median([m["predicted_nnz_pre_mask"] / (m["shape"][0] * m["shape"][1]) for m in pm]))
cfgp = ROOT / f"unified/configs/{tf}.yaml"
cfg = load_config(cfgp if cfgp.exists() else ROOT / "unified/configs/default.yaml")
m = cfg["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_cells = sum(1 for _ in open(w / "seqs/barcodes.tsv"))
ck = torch.load(w / "model/ckpt.pt", map_location=device)
ckm = (ck.get("config") or {}).get("model") or {}
assert not bool(ckm.get("sparsity_aware", m.get("sparsity_aware", False)))
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
h5 = h5py.File(w / "seqs/seqs.h5", "r")
X = h5["X"]
mm = np.asarray(h5["mm_row_idx"])
cells = None
cidx = np.arange(n_cells)
if cap and n_cells > cap:
    cidx = np.sort(np.random.default_rng(7).choice(n_cells, cap, replace=False))
    cells = torch.from_numpy(cidx).to(device)

# rule b cutoff: sample like 02_impute pass 1
cr = json.loads((w / "seqs/chrom_ranges.json").read_text())
rng = np.random.default_rng(0)
wts = np.array([c["n_bins"] for c in cr], float); wts /= wts.sum()
target = max(1, 4_000_000 // n_cells)
vals, got = [], 0
with torch.no_grad():
    while got < target:
        c = cr[int(rng.choice(len(cr), p=wts))]
        if c["n_bins"] == 0:
            continue
        lo = c["start_idx"] + int(rng.integers(0, c["n_bins"]))
        hi = min(c["end_idx"], lo + min(512, target - got + 16))
        y = model.predict_chunk(model.encode_bins(torch.from_numpy(X[lo:hi].astype(np.float32)).to(device)), cell_idx=cells)
        vals.append(y.cpu().numpy().ravel()); got += hi - lo
sv = np.concatenate(vals)
CUT_B = float(np.quantile(sv, 1.0 - FRAC_B))
print(tf, "cut_a", CUT_A, "frac_b", FRAC_B, "cut_b", CUT_B, "current", json.loads((w / "impute/meta.json").read_text())["threshold"], flush=True)

# raw positives for the audit bins
OC = ROOT / "results_293T_analysis/fig2_qc/imputed_audit/continuous"
want = np.load(OC / "bins_mm.npy")
raw = sio.mmread(gzip.open(w / "mm/matrix.mtx.gz")).tocsr()[:, cidx]
rawpos = (raw[want] > 0)
pos = np.full(int(mm.max()) + 1, -1, np.int64)
pos[mm] = np.arange(len(mm))
loc = np.where(want <= mm.max(), pos[np.minimum(want, mm.max())], -1)
ok = np.flatnonzero(loc >= 0)
ok = ok[np.argsort(loc[ok])]
nc = len(cidx)
out = {k: np.full(len(want), np.nan, np.float32) for k in ("a_union", "b_union", "raw_only", "a_pred", "b_pred")}
CH = 512
with torch.no_grad():
    for i in range(0, len(ok), CH):
        rows = ok[i:i + CH]
        y = model.predict_chunk(model.encode_bins(torch.from_numpy(X[loc[rows]].astype(np.float32)).to(device)), cell_idx=cells)
        y = y.cpu().numpy()
        rp = rawpos[rows].toarray()
        pa, pb = y > CUT_A, y > CUT_B
        out["a_pred"][rows] = pa.mean(1)
        out["b_pred"][rows] = pb.mean(1)
        out["a_union"][rows] = (pa | rp).mean(1)
        out["b_union"][rows] = (pb | rp).mean(1)
        out["raw_only"][rows] = rp.mean(1)
miss = np.setdiff1d(np.arange(len(want)), ok)
out["raw_only"][miss] = np.asarray(rawpos[miss].mean(1)).ravel()
for k in ("a_union", "b_union"):
    out[k][miss] = out["raw_only"][miss]
np.savez(w / "predict/rethreshold_audit_bins.npz", bins_mm=want, cut_a=CUT_A, cut_b=CUT_B, frac_b=FRAC_B, n_cells_used=nc, **out)
print("done", tf, nc)
