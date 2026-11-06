# Fetch canonical UniProt entries (JSON) and AlphaFold DB per-residue pLDDT for the panel proteins
import json, sys, os, time, urllib.request, urllib.parse

genes_file, out = sys.argv[1:3]
tsv = sys.argv[3] if len(sys.argv) > 3 else "proteins.tsv"
os.makedirs(f"{out}/uniprot", exist_ok=True)
os.makedirs(f"{out}/afdb", exist_ok=True)
ALIAS = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}

def get(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2 ** (i + 1))

panel = [g.strip().upper() for g in open(genes_file) if g.strip()]
rows = []
for p in panel:
    for g in ALIAS.get(p, [p]):
        q = urllib.parse.quote(f"gene_exact:{g} AND organism_id:9606 AND reviewed:true")
        r = json.loads(get(f"https://rest.uniprot.org/uniprotkb/search?query={q}&format=json&size=5"))["results"]
        r = [x for x in r if any(gn["geneName"]["value"] == g for gn in x.get("genes", []) if "geneName" in gn)] or r
        acc = r[0]["primaryAccession"]
        d = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
        json.dump(d, open(f"{out}/uniprot/{g}.json", "w"))
        L = d["sequence"]["length"]
        # AFDB: per-residue pLDDT; long proteins may be split into fragments or absent
        af = "none"
        try:
            meta = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"))
            m = [x for x in meta if x.get("uniprotAccession") == acc and x.get("uniprotStart", 1) == 1] or meta
            conf = json.loads(get(m[0]["plddtDocUrl"] if "plddtDocUrl" in m[0] else m[0]["pdbUrl"].replace("model_v", "confidence_v").replace(".pdb", ".json")))
            json.dump({"start": m[0].get("uniprotStart", 1), "end": m[0].get("uniprotEnd", L),
                       "seq": m[0].get("uniprotSequence", ""), "plddt": conf["confidenceScore"]}, open(f"{out}/afdb/{g}.json", "w"))
            af = f"{m[0].get('uniprotStart', 1)}-{m[0].get('uniprotEnd', L)}"
        except Exception as e:
            af = f"none ({type(e).__name__})"
        rows.append((p, g, acc, L, af))
        print(p, g, acc, L, af, flush=True)

with open(f"{out}/{tsv}", "w") as f:
    f.write("panel_name\tgene\tuniprot\tlength\tafdb_range\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
