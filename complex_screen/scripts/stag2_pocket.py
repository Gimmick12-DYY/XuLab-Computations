# STAG2-pocket check for Boltz screen hits involving STAG2.
# Pocket = STAG2 residues (UniProt numbering) within 5 A of the CTCF YDF-motif segment (CTCF 222-232) in the calibration
# models p01 (CTCF 180-265 + STAG2 80-1060) and p02 (+ RAD21), top-ranked sample. For each STAG2 hit (top CIF kept by the
# screen), report the STAG2 interface residues, the fraction inside the pocket, and the partner segment that occupies it.
# Writes complex_screen/screen/stag2_pocket.tsv
import glob, json, re, sys
import numpy as np, pandas as pd, gemmi
from scipy.spatial import cKDTree

C = "/work/users/d/y/dyy12/XuLab/complex_screen"
STAG2_START = 80  # screen chain STAG2 = UniProt 80-1060 (contiguous), calibration construct identical
CUT = 5.0

def heavy(ch):
    out = []
    for r in ch:
        for a in r:
            if a.element.name != "H": out.append((r.seqid.num, a.pos.tolist()))
    return out

def contacts(ch_a, ch_b):
    """residue numbers of a within CUT of b, and of b within CUT of a"""
    A, B = heavy(ch_a), heavy(ch_b)
    ta, tb = cKDTree([p for _, p in A]), cKDTree([p for _, p in B])
    ra = {A[i][0] for i, nb in enumerate(tb.query_ball_point([p for _, p in A], CUT)) if nb}
    rb = {B[i][0] for i, nb in enumerate(ta.query_ball_point([p for _, p in B], CUT)) if nb}
    return ra, rb

def seq(ch): return gemmi.one_letter_code([r.name for r in ch])

# ---- pocket from calibration (CTCF YDF segment) ----------------------------------------------------------------------
pocket, pocket_src = set(), []
for pid in ["p01_CTCF_STAG2", "p02_CTCF_STAG2_RAD21"]:
    f = sorted(glob.glob(f"{C}/calib/out/boltz_results_{pid}/predictions/{pid}/{pid}_model_0.cif"))
    if not f: continue
    m = gemmi.read_structure(f[0])[0]
    ctcf, stag2 = m["A"], m["B"]
    seg = gemmi.Chain("Y")
    for r in ctcf:
        if 222 <= r.seqid.num + 179 <= 232: seg.add_residue(r)
    rs, _ = contacts(stag2, seg)
    pocket |= {x + STAG2_START - 1 for x in rs}; pocket_src.append(f"{pid}: {len(rs)} residues")
print("pocket:", len(pocket), "STAG2 residues;", "; ".join(pocket_src))
print("pocket residues (UniProt):", sorted(pocket))

# ---- STAG2 hits ------------------------------------------------------------------------------------------------------------
S = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t")
hits = S[(S.status == "ok") & ((S.chain_a == "STAG2") | (S.chain_b == "STAG2")) & (S.ipsae_max.astype(float) >= 0.1)]
chains = pd.read_csv(f"{C}/chains/chains.tsv", sep="\t").set_index("chain_id")
rows = []
for r in hits.itertuples():
    f = f"{C}/screen/structures/{r.pair_id}.cif"
    try: m = gemmi.read_structure(f)[0]
    except Exception: rows.append(dict(pair_id=r.pair_id, status="no_structure")); continue
    s_id, p_id = ("A", "B") if r.chain_a == "STAG2" else ("B", "A")
    partner = r.chain_b if s_id == "A" else r.chain_a
    rs, rp = contacts(m[s_id], m[p_id])
    rs_u = {x + STAG2_START - 1 for x in rs}
    inpk = rs_u & pocket
    # partner residues touching the pocket, mapped back to UniProt via the chain segments (linkers excluded)
    segs = [tuple(map(int, x.split("-"))) for x in chains.loc[partner, "segments"].split(",")]
    idx2u, k = {}, 0
    for i, (a, b) in enumerate(segs):
        for u in range(a, b + 1): k += 1; idx2u[k] = u
        k += 10 if i < len(segs) - 1 else 0
    pk_chain = gemmi.Chain("P")
    for res in m[s_id]:
        if res.seqid.num + STAG2_START - 1 in pocket: pk_chain.add_residue(res)
    _, rp_pk = contacts(pk_chain, m[p_id]) if len(pk_chain) else (set(), set())
    pseq = seq(m[p_id])
    pk_u = sorted(idx2u.get(x) for x in rp_pk if x in idx2u)
    motif = ""
    if rp_pk:
        lo, hi = min(rp_pk), max(rp_pk)
        motif = pseq[max(0, lo - 3):hi + 2]
    rows.append(dict(pair_id=r.pair_id, partner=partner, iptm_top=r.iptm_top, ipsae_max=r.ipsae_max, hit=r.hit,
                     stag2_interface=len(rs_u), in_pocket=len(inpk), pocket_frac=len(inpk) / max(len(rs_u), 1),
                     partner_interface=len(rp), partner_res_in_pocket=",".join(map(str, pk_u)) or ".",
                     partner_pocket_segment=motif or ".",
                     stag2_pocket_class=len(inpk) >= 5 and len(inpk) / max(len(rs_u), 1) >= 0.3))
T = pd.DataFrame(rows)
T.to_csv(f"{C}/screen/stag2_pocket.tsv", sep="\t", index=False, float_format="%.3f")
json.dump(dict(pocket=sorted(pocket), source=pocket_src, cutoff_A=CUT), open(f"{C}/screen/stag2_pocket_def.json", "w"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(T.round(3).to_string(index=False))
