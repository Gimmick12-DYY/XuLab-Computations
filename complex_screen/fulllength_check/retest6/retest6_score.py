# Compare trimmed (screen) vs full-length Boltz-2 models for the strong non-STAG2 hits:
# ipTM (top-ranked sample), ipSAE (best of 5), tier, interface residues (UniProt numbering, 5 A heavy-atom contacts;
# trimmed numbering mapped back through the chain segments), Jaccard per chain. Extra seeds (2, 3) for the two pairs run
# with them. Verdict: robust = full-length tier >= weak AND interface Jaccard >= 0.3 on both chains; construct-dependent =
# full-length ipSAE < 0.1; otherwise "changed interface". Writes fulllength_check/fulllength_scores.tsv
import glob, json, os
import numpy as np, pandas as pd, gemmi
from scipy.spatial import cKDTree
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/fulllength_check/retest6"
ch = pd.read_csv(f"{C}/chains/chains.tsv", sep="\t").set_index("chain_id")
S = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t").set_index("pair_id")

def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; n = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1); best = max(best, float(s.max()))
    return best
def tier(ip, ps): return "strong" if (ps >= 0.3 and ip >= 0.4) else ("weak" if ps >= 0.1 else "none")
def idx2u(t):
    segs = [tuple(map(int, x.split("-"))) for x in ch.loc[t, "segments"].split(",")]; m, k = {}, 0
    for i, (a, b) in enumerate(segs):
        for u in range(a, b + 1): k += 1; m[k] = u
        k += 10 if i < len(segs) - 1 else 0
    return m
def iface(cif, ma=None, mb=None):
    m = gemmi.read_structure(cif)[0]
    A = [(r.seqid.num, a.pos.tolist()) for r in m["A"] for a in r if a.element.name != "H"]
    B = [(r.seqid.num, a.pos.tolist()) for r in m["B"] for a in r if a.element.name != "H"]
    tA, tB = cKDTree([p for _, p in A]), cKDTree([p for _, p in B])
    ra = {A[i][0] for i, nb in enumerate(tB.query_ball_point([p for _, p in A], 5)) if nb}
    rb = {B[i][0] for i, nb in enumerate(tA.query_ball_point([p for _, p in B], 5)) if nb}
    if ma: ra = {ma[x] for x in ra if x in ma}
    if mb: rb = {mb[x] for x in rb if x in mb}
    return ra, rb
def jac(x, y): return len(x & y) / len(x | y) if (x | y) else np.nan
def full_run(outdir, pid):
    d = f"{outdir}/boltz_results_inputs*/predictions/{pid}_full"
    conf = sorted(glob.glob(f"{d}/confidence_{pid}_full_model_*.json"))
    if not conf: return None
    dd = os.path.dirname(conf[0])
    seqs = [l.split(":", 1)[1].strip() for l in open(f"{O}/inputs/{pid}_full.yaml") if "sequence:" in l]
    ip, ps, cs = [], [], []
    for f in conf:
        k = f.split("_model_")[1][:-5]; c = json.load(open(f))
        ip.append(c["pair_chains_iptm"]["0"]["1"]); cs.append(c["confidence_score"])
        ps.append(ipsae(np.load(f"{dd}/pae_{pid}_full_model_{k}.npz")["pae"], len(seqs[0]), len(seqs[1])))
    t = int(np.argmax(cs))
    return dict(iptm=ip[t], ipsae=max(ps), tier=tier(ip[t], max(ps)), cif=f"{dd}/{pid}_full_model_{t}.cif")

rows = []
for y in sorted(glob.glob(f"{O}/inputs/*_full.yaml")):
    pid = os.path.basename(y)[:-10]; a, b = pid.split("__")
    r = dict(pair=pid, tokens_trimmed=int(S.loc[pid, "n_tokens"]) if pid in S.index else None,
             tokens_full=sum(len(l.split(":", 1)[1].strip()) for l in open(y) if "sequence:" in l))
    if pid in S.index:
        r.update(trim_iptm=S.loc[pid, "iptm_top"], trim_ipsae=S.loc[pid, "ipsae_max"], trim_tier=S.loc[pid, "hit"],
                 hub_partner=S.loc[pid, "hub_partner"] if "hub_partner" in S.columns else ".")
    F1 = full_run(f"{O}/out_seed1", pid)
    if F1 is None: r["full_tier"] = "not run / failed"; rows.append(r); continue
    r.update(full_iptm=F1["iptm"], full_ipsae=F1["ipsae"], full_tier=F1["tier"])
    for sd in (2, 3):
        Fx = full_run(f"{O}/out_seed{sd}", pid)
        if Fx: r[f"full_seed{sd}"] = f"{Fx['iptm']:.2f}/{Fx['ipsae']:.2f} {Fx['tier']}"
    tcif = f"{C}/screen/structures/{pid}.cif"
    if os.path.exists(tcif):
        ta, tb = iface(tcif, idx2u(a), idx2u(b)); fa, fb = iface(F1["cif"])
        r.update(iface_trim_a=len(ta), iface_full_a=len(fa), jaccard_a=jac(ta, fa), iface_trim_b=len(tb), iface_full_b=len(fb), jaccard_b=jac(tb, fb))
    if F1["ipsae"] < 0.1: v = "construct-dependent"
    elif r.get("jaccard_a", 0) >= 0.3 and r.get("jaccard_b", 0) >= 0.3: v = "robust"
    else: v = "changed interface"
    r["verdict"] = v; rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{O}/fulllength_scores.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
print(T.verdict.value_counts().to_dict() if "verdict" in T else "no results")
