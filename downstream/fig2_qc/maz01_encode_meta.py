"""MAZ reference audit, step 1: pull ENCODE metadata + QC for all human MAZ TF ChIP-seq experiments.
Output: results_293T_analysis/maz_ref_audit/encode_maz_qc.tsv, encode_maz_audits.tsv, encode_maz_files.tsv
"""
import json, sys, urllib.request
from pathlib import Path
import pandas as pd

OUT = Path("/work/users/d/y/dyy12/XuLab/results_293T_analysis/maz_ref_audit"); OUT.mkdir(parents=True, exist_ok=True)
E = "https://www.encodeproject.org"
def get(u):
    r = urllib.request.Request(E + u, headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))

srch = get("/search/?type=Experiment&target.label=MAZ&assay_title=TF+ChIP-seq&status=released&format=json&limit=all")
rows, aud, files = [], [], []
for hit in [h for h in srch["@graph"] if h["accession"] not in ("ENCSR000ESL", "ENCSR000EQT")]:  # drop mouse MEL/CH12
    acc = hit["accession"]
    e = get(f"/experiments/{acc}/?format=json&frame=embedded")
    reps = e.get("replicates", [])
    ab = sorted({r["antibody"]["accession"] for r in reps if r.get("antibody")})
    lib = reps[0]["library"]["biosample"] if reps else {}
    gm = [g.get("description", "") + " | " + g.get("method", "") + " | " + g.get("category", "") for g in lib.get("applied_modifications", lib.get("genetic_modifications", []))]
    for cat, items in get(f"/experiments/{acc}/?format=json&frame=object&datastore=database").get("audit", hit.get("audit", {})).items():
        for a in items:
            aud.append(dict(exp=acc, level=cat, category=a["category"], detail=a.get("detail", "")[:600]))
    # GRCh38 analysis files
    for f in e["files"]:
        if f.get("assembly") != "GRCh38" or f.get("status") != "released":
            continue
        fan = [x if isinstance(x, str) else x.get("@id") for x in f.get("analyses", [])]
        if e.get("default_analysis") not in fan:  # keep the default (ENCODE4) analysis only
            continue
        qms = f.get("quality_metrics", [])
        rec = dict(exp=acc, cell=e["biosample_ontology"]["term_name"], file=f["accession"], output_type=f["output_type"],
                   file_type=f["file_type"], bio_rep=",".join(map(str, f.get("biological_replicates", []))),
                   pipeline_default=f.get("preferred_default", False), href=E + f["href"], read_length=f.get("read_length"), run_type=f.get("run_type"), analysis=e.get("default_analysis"))
        for q in qms:
            if isinstance(q, str):
                try: q = get(q + "?format=json")
                except Exception: continue
            for k in ("frip", "NSC", "RSC", "total_reads", "mapped_reads", "NRF", "PBC1", "PBC2", "rescue_ratio",
                      "self_consistency_ratio", "reproducibility", "N_optimal", "N_conservative", "Np", "Nt", "N1", "N2",
                      "total", "mapped", "paired", "read1", "read2", "duplicate_reads", "pct_duplicate_reads",
                      "processing_stage", "mapped_pct"):
                if k in q:
                    rec.setdefault(f"{q['@type'][0]}:{k}", q[k])
        files.append(rec)
    run_types = sorted({f.get("run_type", "") for f in e["files"] if f.get("file_type") == "fastq"})
    rows.append(dict(exp=acc, run_type=",".join(run_types), cell=e["biosample_ontology"]["term_name"], lab=e["lab"]["title"], released=e.get("date_released"),
                     antibody=",".join(ab), n_bioreps=len({r["biological_replicate_number"] for r in reps}),
                     description=e.get("description", ""), modification=" ; ".join(gm),
                     controls=",".join(c["accession"] for c in e.get("possible_controls", [])),
                     audit_ERROR=sum(1 for a in aud if a["exp"] == acc and a["level"] == "ERROR"),
                     audit_NOT_COMPLIANT=sum(1 for a in aud if a["exp"] == acc and a["level"] == "NOT_COMPLIANT"),
                     audit_WARNING=sum(1 for a in aud if a["exp"] == acc and a["level"] == "WARNING")))
    print(acc, file=sys.stderr)
pd.DataFrame(rows).to_csv(OUT / "encode_maz_experiments.tsv", sep="\t", index=False)
pd.DataFrame(aud).to_csv(OUT / "encode_maz_audits.tsv", sep="\t", index=False)
pd.DataFrame(files).to_csv(OUT / "encode_maz_files.tsv", sep="\t", index=False)
# antibodies
ab_rows = []
for ab in sorted({a for r in rows for a in r["antibody"].split(",") if a}):
    a = get(f"/antibodies/{ab}/?format=json&frame=embedded")
    ab_rows.append(dict(antibody=ab, source=a.get("source", {}).get("title", a.get("source")), product=a.get("product_id"),
                        targets=",".join(t.get("label", str(t)) if isinstance(t, dict) else str(t) for t in a.get("targets", [])),
                        status=a.get("lot_reviews", [{}])[0].get("status") if a.get("lot_reviews") else a.get("status"),
                        lot_reviews=" ; ".join(f"{','.join(b.get('label','') if isinstance(b,dict) else str(b) for b in lr.get('biosample_term_name',[]) ) if False else ''}{lr.get('status')}:{lr.get('detail','')[:150]}" for lr in a.get("lot_reviews", []))))
pd.DataFrame(ab_rows).to_csv(OUT / "encode_maz_antibodies.tsv", sep="\t", index=False)
