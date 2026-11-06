# Manifest v3: rehashes the v1 file list (unchanged) and adds the cis-coordinated outputs (R1 v3 tables, propagation). The B0 arms have their own manifest. The v1 manifest is left as it is.
import glob, hashlib
X = "/work/users/d/y/dyy12/XuLab"; R1 = f"{X}/regulatome/r1_tftf"; P = f"{X}/regulatome/r2_targets/propagation_ciscoord_v3"
v1 = [l.split("  ", 1) for l in open(f"{R1}/v2_final/frozen_manifest.sha256").read().splitlines()]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); bad = [p for s, p in v1 if h(p) != s]; assert not bad, f"v1 files changed: {bad[:5]}"
fs = [p for _, p in v1] + [f"{R1}/v2_final/frozen_manifest.sha256"] + sorted(glob.glob(f"{R1}/v2_final_v3/r1_*") + glob.glob(f"{R1}/v2_final_v3/cis_*")) + sorted(glob.glob(f"{P}/*.tsv*") + glob.glob(f"{P}/inferred_lists/*.tsv"))
L = "\n".join(f"{h(p)}  {p}" for p in fs) + "\n"; open(f"{R1}/v2_final_v3/frozen_manifest_v3.sha256", "w").write(L); print("v1 files verified unchanged:", len(v1), "| v3 files:", len(fs), "| sha256 of manifest v3:", hashlib.sha256(L.encode()).hexdigest())
