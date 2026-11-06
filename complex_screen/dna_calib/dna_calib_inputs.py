# Boltz-2 DNA calibration inputs: 8BYX (HOXB13 homodimer on DNA) and 1NKP (MYC-MAX on E-box); sequences from the PDB entities,
# one unpaired ColabFold MSA per protein construct
import os, sys
from boltz.data.msa.mmseqs2 import run_mmseqs2
D = sys.argv[1]
HOXB13 = "GRKKRIPYSKGQLRELEREYAANKFITKDKRRKISAATSLSERQITIWFQNRRVKEKKVLAKV"
MYC = "GHMNVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLGGC"
MAX = "DKRAHHNALERKRRDHIKDSFHSLRDSVPSLQGEKASRAQILDKATEYIQYMRRKNHTHQQDIDDLKRQNALLEQQVRALGGC"
msa = {}
for name, s in [("HOXB13_8BYX", HOXB13), ("MYC_1NKP", MYC), ("MAX_1NKP", MAX)]:
    dst = f"{D}/msa/{name}.csv"
    if not os.path.exists(dst):
        a3m = run_mmseqs2(s, f"{D}/msa/tmp_{name}", use_env=True, use_pairing=False)
        a3m = a3m[0] if isinstance(a3m, list) else a3m
        seqs = [l for l in a3m.strip().splitlines() if not l.startswith(">")]
        with open(dst, "w") as f: f.write("key,sequence\n" + "\n".join(f"-1,{x}" for x in seqs) + "\n")
    msa[name] = dst
with open(f"{D}/inputs/8BYX_HOXB13_dimer_DNA.yaml", "w") as f:
    f.write(f"""version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: {HOXB13}
      msa: {msa['HOXB13_8BYX']}
  - dna:
      id: C
      sequence: GGACCAATAAAATACAAC
  - dna:
      id: D
      sequence: GTTGTATTTTATTGGTCC
""")
with open(f"{D}/inputs/1NKP_MYC_MAX_DNA.yaml", "w") as f:
    f.write(f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {MYC}
      msa: {msa['MYC_1NKP']}
  - protein:
      id: B
      sequence: {MAX}
      msa: {msa['MAX_1NKP']}
  - dna:
      id: [C, D]
      sequence: CGAGTAGCACGTGCTACTC
""")
print("ok", msa)
