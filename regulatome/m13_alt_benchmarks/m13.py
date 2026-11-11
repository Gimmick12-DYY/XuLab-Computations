# m13_alt_benchmarks (SPEC.md): coverage of alternative benchmark data for the 79 panel TFs. Metadata and public lists only; no comparison with our binding.
# usage: m13.py fetch  (login node, network) -> XuLab/data/external/... + sources.tsv ; m13.py build -> coverage.tsv, coverage_summary.tsv
import datetime, hashlib, json, os, re, subprocess, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m13_alt_benchmarks"; EXT = f"{X}/data/external"; LAB = "coverage only; no comparison with our binding"
for d in ("tf_targets", "chipatlas", "geo", "encode", "knocktf", "replogle"): os.makedirs(f"{EXT}/{d}", exist_ok=True)
SRC = {
    "tf_targets/collectri_omnipath.tsv": "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1&fields=is_stimulation,is_inhibition,consensus_direction,consensus_stimulation,consensus_inhibition&organisms=9606&license=academic",
    "tf_targets/dorothea_ABC_omnipath.tsv": "https://omnipathdb.org/interactions?datasets=dorothea&dorothea_levels=A,B,C&genesymbols=1&fields=is_stimulation,is_inhibition,consensus_direction,dorothea_level&organisms=9606&license=academic",
    "tf_targets/trrust_rawdata.human.tsv": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    "chipatlas/experimentList.tab": "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab",
    "geo/GSE78099_gsm_full.txt": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78099&targ=gsm&form=text&view=full",
    "encode/tf_chipseq_experiments.json": "https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&limit=all&format=json&field=accession&field=target.label&field=biosample_ontology.term_name&field=biosample_summary&field=files.file_type&field=files.output_type&field=files.assembly",
    "encode/perturb_rnaseq_experiments.json": "https://www.encodeproject.org/search/?type=Experiment&assay_title=shRNA+RNA-seq&assay_title=CRISPRi+RNA-seq&assay_title=CRISPR+RNA-seq&status=released&limit=all&format=json&field=accession&field=assay_title&field=target.label&field=biosample_ontology.term_name",
    "knocktf/knocktf_v2_main_human.txt": "http://www.licpathway.net/KnockTFv2/public/download_anno/knocktf_v2_main_human.txt"}
REPL = {"K562_gwps": "https://ndownloader.figshare.com/files/35773217", "K562_essential": "https://ndownloader.figshare.com/files/35780870", "RPE1": "https://ndownloader.figshare.com/files/35775512"}
def sha(f): h = hashlib.sha256(); h.update(open(f, "rb").read()); return h.hexdigest()
if sys.argv[1] == "fetch":
    rows = []; today = datetime.date.today().isoformat()
    for rel, url in SRC.items():
        f = f"{EXT}/{rel}"; r = subprocess.run(["curl", "-sSL", "--max-time", "1800", "--retry", "3", "-o", f, url], capture_output=True, text=True); ok = r.returncode == 0 and os.path.exists(f) and os.path.getsize(f) > 0
        rows.append(dict(file=f, url=url, date=today, ok=ok, bytes=os.path.getsize(f) if os.path.exists(f) else 0, sha256=sha(f) if ok else "", note=r.stderr.strip()[:200])); print(rel, ok, flush=True)
    # Replogle 2022: perturbation lists only, read from the obs table of the pseudobulk h5ad over HTTP (the expression matrix X is not read or stored)
    import fsspec, h5py
    for k, url in REPL.items():
        try:
            fh = fsspec.open(url, block_size=2 ** 20).open(); h = h5py.File(fh, "r"); ob = h["obs"]; key = ob.attrs.get("_index", "_index"); key = key.decode() if isinstance(key, bytes) else key; node = ob[key]
            if isinstance(node, h5py.Group): cats = node["categories"][()]; idx = cats[node["codes"][()]]
            else: idx = node[()]
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(idx)]; f = f"{EXT}/replogle/{k}_perturbation_obs_index.txt"; open(f, "w").write("\n".join(names) + "\n")
            rows.append(dict(file=f, url=url + " (obs only via HTTP range reads; X not read)", date=today, ok=True, bytes=os.path.getsize(f), sha256=sha(f), note=f"{len(names)} obs rows")); print(k, len(names), names[:3], flush=True)
        except Exception as ex: rows.append(dict(file="", url=url, date=today, ok=False, bytes=0, sha256="", note=str(ex)[:200])); print(k, "failed", ex, flush=True)
    pd.DataFrame(rows).to_csv(f"{O}/sources.tsv", sep="\t", index=False)
else:
    tfs = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; GENES = {t: [t] for t in tfs}; GENES["HP1"] = ["CBX1", "CBX3", "CBX5"]; GENES["ZUFSP"] = ["ZUP1"]
    def gs(t): return GENES[t]
    FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF")
    TEST = set(pd.read_csv(f"{RG}/m10_promoter_regulome/F1_testable.tsv", sep="\t").query("arm == 'arm2'").tf); UNI = set(np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str))
    T = pd.DataFrame(index=tfs); T["cells"] = [FL.cells.get(t, np.nan) for t in tfs]; T["depth_tertile"] = [FL.tertile.get(t, "") for t in tfs]; T["claim_tier_d_ge_0.04"] = [bool(FL.d.get(t, 0) >= .04) for t in tfs]
    T["kd_ceiling"] = [FT.ceiling.get(t, np.nan) for t in tfs]; T["benchmark_testable_m10"] = [t in TEST for t in tfs]; T["genes"] = [",".join(gs(t)) for t in tfs]
    # ---- ChIP: ENCODE
    def ok(f): return os.path.exists(f) and os.path.getsize(f) > 0
    f = f"{EXT}/encode/tf_chipseq_experiments.json"
    if ok(f):
        G = json.load(open(f))["@graph"]; E = []
        for g in G:
            tl = (g.get("target") or {}).get("label", ""); bs = (g.get("biosample_ontology") or {}).get("term_name", ""); fl = g.get("files", []) or []
            pk = any(x.get("file_type") == "bed narrowPeak" and x.get("assembly") == "GRCh38" for x in fl); summ = g.get("biosample_summary", "") or ""
            tag = "eGFP" if "eGFP" in summ or "GFP" in summ else ("HA" if "HA" in summ.split() or "HA-tagged" in summ else ("FLAG" if "FLAG" in summ else "endogenous/unspecified"))
            E.append(dict(acc=g.get("accession"), target=tl.upper(), biosample=bs, peaks_grch38=pk, tag=tag))
        E = pd.DataFrame(E)
        for lab, sel in (("HEK293T", E.biosample == "HEK293T"), ("HEK293", E.biosample == "HEK293"), ("other", ~E.biosample.isin(["HEK293", "HEK293T"]))):
            T[f"encode_chip_{lab}_n"] = [int((sel & E.target.isin(gs(t))).sum()) for t in tfs]
            if lab != "other": T[f"encode_chip_{lab}_acc_tag_peak"] = [";".join(f"{r.acc}({r.tag},{'peaks' if r.peaks_grch38 else 'no GRCh38 peaks'})" for r in E[sel & E.target.isin(gs(t))].itertuples()) for t in tfs]
    else: T["encode_chip_HEK293T_n"] = T["encode_chip_HEK293_n"] = T["encode_chip_other_n"] = np.nan
    # ---- ChIP-Atlas
    f = f"{EXT}/chipatlas/experimentList.tab"
    if ok(f):
        rows_ = []
        with open(f, errors="ignore") as fh:
            for line in fh:
                x = line.rstrip("\n").split("\t")
                if len(x) < 6 or x[1] != "hg38": continue
                rows_.append((x[0], x[3], x[5], " ".join(x[6:]).upper()))
        CA = pd.DataFrame(rows_, columns=["id", "antigen", "cell", "meta"]); CA["A"] = CA.antigen.str.upper().str.replace(r"^(EGFP|GFP|HA|FLAG)-", "", regex=True)
        cell = CA.cell.fillna(""); hek = cell.isin(["293", "HEK293-T-REx"]); t_ = CA.meta.str.contains(r"293 ?-?T(?![A-Z0-9-]*REX)", regex=True)
        is293t = hek & t_; is293 = hek & ~t_
        # ChIP-Atlas files HEK293 and HEK293T under cell type "293"; HEK293T is assigned only where the sample title/metadata says 293T
        for lab, sel in (("HEK293T", is293t), ("HEK293", is293), ("other", ~hek)):
            T[f"chipatlas_{lab}_n"] = [int((sel & CA.A.isin(gs(t))).sum()) for t in tfs]
        T["chipatlas_HEK293T_ids"] = [";".join(CA[is293t & CA.A.isin(gs(t))].id.head(10)) for t in tfs]; T["chipatlas_HEK293_ids"] = [";".join(CA[is293 & CA.A.isin(gs(t))].id.head(10)) for t in tfs]
    # ---- GSE78099 (KZFP ChIP-exo, HEK293T)
    f = f"{EXT}/geo/GSE78099_gsm_full.txt"
    if ok(f):
        txt = open(f, errors="ignore").read().split("^SAMPLE = ")[1:]; S = []
        for blk in txt:
            gsm = blk.split("\n")[0].strip(); title = re.search(r"!Sample_title = (.*)", blk); sup = re.findall(r"!Sample_supplementary_file_\d+ = (.*)", blk); S.append(dict(gsm=gsm, title=title.group(1) if title else "", n_supp=sum(1 for s in sup if s.strip() != "NONE"), supp=";".join(os.path.basename(s) for s in sup)))
        S = pd.DataFrame(S); toks = S.title.str.upper().str.split(r"[^A-Z0-9]+")
        T["GSE78099_HEK293T_n"] = [int(sum(any(g in tk for g in gs(t)) for tk in toks)) for t in tfs]
        T["GSE78099_gsm_supp"] = [";".join(f"{r.gsm}({r.n_supp} files)" for r, tk in zip(S.itertuples(), toks) if any(g in tk for g in gs(t))) for t in tfs]
    # ---- on Longleaf already
    bulk = pd.read_csv(f"{X}/downstream/tracks/bulk_chipseq/manifest.tsv", sep="\t"); T["longleaf_bulk_chip"] = [";".join(f"{r.tf}({'K562' if 'K562' in r.bam else 'HEK293'})" for r in bulk.itertuples() if r.tf == t) for t in tfs]
    T.loc["CTCF", "longleaf_bulk_chip"] += ";CTCF(HEK293T p-value track)"
    cn = set(os.listdir("/users/x/u/xujie/HEK293T_SA")) if os.path.isdir("/users/x/u/xujie/HEK293T_SA") else set(); T["longleaf_chromnitron"] = [t in cn for t in tfs]
    # ---- motifs
    MW = pd.read_csv(f"{RG}/m9_abc_diagnosis/B_motif_workbook.tsv", sep="\t").set_index("tf"); T["hocomoco_v11_motif"] = [MW.hocomoco_v11_motif.get(t, "not testable") for t in tfs]
    def meme_names(f): return [l.split()[1:] for l in open(f) if l.startswith("MOTIF")]
    h12 = meme_names(f"{X}/motif_analysis/cache/motifdb/H12CORE_meme_format.meme"); jas = meme_names(f"{X}/motif_analysis/cache/motifdb/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")
    T["hocomoco_v12_core_n"] = [sum(1 for m in h12 if any(m[0].upper().split(".")[0] == g for g in gs(t))) for t in tfs]
    T["jaspar2024_n"] = [sum(1 for m in jas if len(m) > 1 and any(g in m[1].upper().split("::") for g in gs(t))) for t in tfs]
    WB = pd.read_excel(f"{X}/motif_analysis/TF1000cells_motifs.xlsx", sheet_name="TF1000cells_motifs").set_index("TF"); pc = [c for c in WB.columns if "motif for" in c]
    T["workbook_own_motif_methods_with_p"] = [int(WB.loc[t, pc].notna().sum()) if t in WB.index else 0 for t in tfs]; T["workbook_best_p"] = [float(WB.loc[t, pc].min()) if t in WB.index and WB.loc[t, pc].notna().any() else np.nan for t in tfs]
    # ---- curated targets
    for nm, f, sc, tc in (("collectri", f"{EXT}/tf_targets/collectri_omnipath.tsv", "source_genesymbol", "target_genesymbol"), ("dorothea_ABC", f"{EXT}/tf_targets/dorothea_ABC_omnipath.tsv", "source_genesymbol", "target_genesymbol")):
        if ok(f):
            D = pd.read_csv(f, sep="\t"); D["signed"] = (D.get("is_stimulation", 0).astype(int) + D.get("is_inhibition", 0).astype(int)) == 1
            T[f"{nm}_targets"] = [int(D[D[sc].isin(gs(t))][tc].nunique()) for t in tfs]; T[f"{nm}_in_universe"] = [len(set(D[D[sc].isin(gs(t))][tc]) & UNI) for t in tfs]; T[f"{nm}_signed"] = [int(D[D[sc].isin(gs(t)) & D.signed][tc].nunique()) for t in tfs]
    f = f"{EXT}/tf_targets/trrust_rawdata.human.tsv"
    if ok(f):
        D = pd.read_csv(f, sep="\t", header=None, names=["tf", "target", "mode", "pmid"]); T["trrust_targets"] = [int(D[D.tf.isin(gs(t))].target.nunique()) for t in tfs]; T["trrust_in_universe"] = [len(set(D[D.tf.isin(gs(t))].target) & UNI) for t in tfs]
        T["trrust_signed"] = [int(D[D.tf.isin(gs(t)) & D["mode"].isin(["Activation", "Repression"])].target.nunique()) for t in tfs]
    # ---- other perturbations
    for k in ("K562_gwps", "K562_essential", "RPE1"):
        f = f"{EXT}/replogle/{k}_perturbation_obs_index.txt"
        if ok(f):
            idx = [l.strip() for l in open(f) if l.strip()]; tok = set(re.split(r"[_|]", x)[1] if re.match(r"^\d+_", x) else re.split(r"[_|]", x)[0] for x in idx); tok |= {p for x in idx for p in re.split(r"[_|]", x)}
            T[f"replogle_{k}"] = [any(g in tok for g in gs(t)) for t in tfs]
        else: T[f"replogle_{k}"] = np.nan
    f = f"{EXT}/knocktf/knocktf_v2_main_human.txt"
    if ok(f):  # per-gene differential expression table; a dataset = a Sample_ID; cell type is not in this file
        K = pd.read_csv(f, sep="\t", usecols=["Sample_ID", "TF"], dtype=str).drop_duplicates(); KT = K.TF.str.upper().str.strip()
        T["knocktf_datasets"] = [int(K.Sample_ID[KT.isin(gs(t))].nunique()) for t in tfs]; T["knocktf_sample_ids"] = [";".join(sorted(K.Sample_ID[KT.isin(gs(t))].unique())[:10]) for t in tfs]
    f = f"{EXT}/encode/perturb_rnaseq_experiments.json"
    if ok(f):
        G = pd.DataFrame([dict(acc=g.get("accession"), assay=g.get("assay_title"), target=(g.get("target") or {}).get("label", "").upper(), bs=(g.get("biosample_ontology") or {}).get("term_name", "")) for g in json.load(open(f))["@graph"]])
        T["encode_perturb_rnaseq"] = [";".join(f"{r.assay}:{r.bs}:{r.acc}" for r in G[G.target.isin(gs(t))].itertuples()) for t in tfs]; T["encode_perturb_rnaseq_n"] = [int(G.target.isin(gs(t)).sum()) for t in tfs]
    # ---- own expression in HEK293T NTC cells
    d = np.load(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); gn = pd.Series(np.arange(len(d["genes"])), index=d["genes"].astype(str)); gn = gn[~gn.index.duplicated(keep="last")]
    T["ntc_detect"] = [";".join(f"{g}:{d['ntc_detect'][gn[g]]:.3f}" if g in gn.index else f"{g}:NA" for g in gs(t)) for t in tfs]; T["ntc_mean_log1pCP10k"] = [";".join(f"{g}:{d['ntc_mean'][gn[g]]:.3f}" if g in gn.index else f"{g}:NA" for g in gs(t)) for t in tfs]
    T.index.name = "tf"; T.insert(0, "label", LAB); T.to_csv(f"{O}/coverage.tsv", sep="\t")
    claim = T.index[T["claim_tier_d_ge_0.04"]]
    def cov(col, cond): s = T.index[cond]; return dict(option=col, tfs_covered=len(s), of_79=79, claim_tier_covered=len(set(s) & set(claim)), of_claim=len(claim), claim_tier_names=",".join(sorted(set(s) & set(claim))), all_names=",".join(sorted(s)))
    rows = []
    for col in [c for c in T.columns if c.endswith("_n") or c.endswith("_targets") or c.endswith("_in_universe") or c == "knocktf_datasets" or c.startswith("replogle_") or c in ("longleaf_chromnitron",)]:
        v = T[col]
        if v.dtype == bool or set(v.dropna().unique()) <= {True, False}: rows.append(cov(col, v.fillna(False).astype(bool).values))
        else: rows.append(cov(col, (pd.to_numeric(v, errors="coerce").fillna(0) > 0).values))
    rows.append(cov("hocomoco_v11_motif", (T.hocomoco_v11_motif.astype(str).isin(["none", "not testable"]) == False).values)); rows.append(cov("longleaf_bulk_chip", (T.longleaf_bulk_chip.fillna("") != "").values))
    pd.DataFrame(rows).to_csv(f"{O}/coverage_summary.tsv", sep="\t", index=False); print(pd.DataFrame(rows)[["option", "tfs_covered", "claim_tier_covered"]].to_string(index=False))
