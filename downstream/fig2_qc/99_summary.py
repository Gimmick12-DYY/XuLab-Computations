#!/usr/bin/env python
"""Merge OUT/summary_parts/*.tsv into OUT/summary_diagonal.tsv (one row per test x arm x variant)."""
import pandas as pd

from fig2_common import OUT

parts = sorted((OUT / "summary_parts").glob("*.tsv"))
df = pd.concat([pd.read_csv(p, sep="\t").assign(script=p.stem) for p in parts], ignore_index=True)
cols = ["test", "arm", "variant", "mean_diag", "diag_minus_offdiag", "own_best_col", "own_best_row", "perm_p", "perm_z"]
df = df[[c for c in cols if c in df] + [c for c in df if c not in cols]]
df.to_csv(OUT / "summary_diagonal.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(df.drop(columns=["script", "n_perm"], errors="ignore").to_string())
