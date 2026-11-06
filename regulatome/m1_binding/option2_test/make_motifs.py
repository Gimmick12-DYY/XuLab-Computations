# One known motif per panel TF (xlsx "Known via" set): HOCOMOCO v12 rank-0 > v11 best > Codebook top1; renamed to the TF
import sys, re
h12, h11, cb, out_meme, out_tsv = sys.argv[1:6]
HOCO = {"CTCF": "CTCF", "E2F3": "E2F3", "FOXJ3": "FOXJ3", "FOXK1": "FOXK1", "MAZ": "MAZ", "MEF2A": "MEF2A", "MEF2D": "MEF2D",
        "NFYA": "NFYA", "NRF1": "NRF1", "ZBTB7A": "ZBT7A", "RBPJ": "SUH", "RELA": "TF65", "ZNF143": "ZN143", "EPAS1": "EPAS1",
        "ETS2": "ETS2", "SOX4": "SOX4", "SREBF2": "SRBP2", "FOXC1": "FOXC1", "TBX3": "TBX3", "HOXA10": "HXA10", "ZNF768": "ZN768",
        "SMARCA1": "SMCA1", "HOXC9": "HXC9", "HOXB8": "HXB8", "MLX": "MLX", "DDIT3": "DDIT3", "E4F1": "E4F1", "HOXB6": "HXB6",
        "ZIC2": "ZIC2", "HES1": "HES1", "ZNF282": "ZN282", "HOXA5": "HXA5", "MSX2": "MSX2", "HOXD10": "HXD10"}
CODEBOOK = ["ZNF507", "MBD3", "ZNF703", "GLYR1", "FLYWCH1", "ZNF367", "ZNF746", "POGK", "ZBTB40", "ZNF606", "ZNF503"]

def blocks(path):
    txt = open(path).read()
    head = txt[:txt.index("MOTIF")]
    parts = re.split(r"(?m)^(?=MOTIF )", txt)[1:]
    return head, {p.split()[1]: p for p in parts}, {p.split()[2] if len(p.split("\n")[0].split()) > 2 else "": p for p in parts}

h12_head, h12_id, _ = blocks(h12)
_, h11_id, _ = blocks(h11)
_, cb_id, cb_alt = blocks(cb)
out, rows = [], []
for tf, hn in HOCO.items():
    k12 = [k for k in h12_id if re.match(rf"^{hn}\.H12CORE\.0\.", k)]
    if k12:
        b, src = h12_id[k12[0]], k12[0]
    else:
        q = {"A": 0, "B": 1, "C": 2, "D": 3}
        k11 = sorted([k for k in h11_id if k.startswith(f"{hn}_HUMAN.H11MO.")], key=lambda k: (int(k.split(".")[2]), q[k.split(".")[3]]))
        if not k11: rows.append((tf, "NOT FOUND")); continue
        b, src = h11_id[k11[0]], k11[0]
    out.append(re.sub(r"^MOTIF \S+( \S+)?", f"MOTIF {tf} {src}", b, count=1)); rows.append((tf, src))
for tf in CODEBOOK:
    b = cb_id.get(tf)
    if b is None: rows.append((tf, "NOT FOUND")); continue
    out.append(re.sub(r"^MOTIF \S+( \S+)?", f"MOTIF {tf} codebook_top1", b, count=1)); rows.append((tf, "codebook_top1:" + b.split()[2]))
open(out_meme, "w").write(h12_head + "".join(out))
with open(out_tsv, "w") as f:
    f.write("tf\tmotif_source\n"); [f.write(f"{a}\t{b}\n") for a, b in rows]
print(len(out), "motifs;", [r for r in rows if r[1] == "NOT FOUND"])
