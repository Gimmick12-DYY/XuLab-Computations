import pandas as pd
X="/work/users/d/y/dyy12/XuLab"
L=[l.split() for l in open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme") if l.startswith("MOTIF ")]
print("codebook artifact motifs:", len(L), "examples:", L[:4])
M=pd.read_csv(f"{X}/complex_screen/pair_motif_composite/motif_tables/pair_shared_motifs_p005.tsv",sep="\t")
names=set(); [names.update(x.strip().split(" [")[0] for x in str(s).split(";") if x.strip() and str(s)!="nan") for s in M.shared_db_all]
ids=set(x[1].upper() for x in L)|set(x[2].upper() for x in L if len(x)>2)
hit=[n for n in names if n.upper() in ids]; print("DB motif names in pair table:", len(names), "| matching an artifact id/name:", len(hit), hit[:10])
db=open(f"{X}/complex_screen/pair_motif_composite/db/known.meme").read(); print("artifact ids present in known.meme:", sum(1 for x in L if (" "+x[1]) in db), "of", len(L))
