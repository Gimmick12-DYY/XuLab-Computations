#!/usr/bin/env python3
"""Rematch HOMER denovo motifs vs merged (H12+JASPAR+Codebook) + artifact DB;
rebuild the giant TF × best-match HTML matrices for motif_pmat and motif_imputed.
"""
from __future__ import annotations

import base64
import html as htmllib
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("/work/users/d/y/dyy12/XuLab")
MA = ROOT / "motif_analysis"
TOMTOM = "/nas/longleaf/rhel9/apps/meme/5.5.7/bin/tomtom"
# Primary rematch for the HTML table: Codebook curated top1 (fast, ~204 motifs).
# Full merged DB (H12+JASPAR+Codebook, 2526) is available for offline rank_motifs.
MEME_DB = MA / "cache/motifdb/codebook_top1.meme"
ART_DB = MA / "cache/motifdb/codebook_artifacts.meme"
HOMER2MEME = MA / "scripts/homer_motifs_to_meme.py"

UNKNOWN = (
    "rbbp4 xpa rbbp7 znf507 nme2 mbd3 znf703 glyr1 safb mtf2 hoxb9 flywch1 dr1 "
    "pa2g4 znf512b znf644 znf367 ezh2 srcap stag2 wiz znf777 znf746 champ1 pogk "
    "thap6 zufsp dnmt1 znf512 zbtb43 adnp trafd1 zbtb40 sox12 akap8l znf581 "
    "neurod4 nr1d2 hp1 znf606 znf79 pcgf6 znf503 setdb1 hoxc4"
).split()


def minify_svg(s: str) -> str:
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s, flags=re.S)
    s = re.sub(r"\s+", " ", s).strip()
    if "viewBox" not in s:
        m = re.search(r'width=["\']([0-9.]+)', s)
        n = re.search(r'height=["\']([0-9.]+)', s)
        if m and n:
            s = s.replace("<svg", f'<svg viewBox="0 0 {m.group(1)} {n.group(1)}"', 1)
    return s


def data_uri(svg: str) -> str:
    b = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b}"


def short_name(target: str) -> str:
    t = target.split("__", 1)[1] if "__" in target else target
    t = t.split(".")[0]
    t = re.sub(r"_HUMAN.*", "", t)
    return t


def parse_tomtom_best(tsv: Path) -> dict[str, tuple[str, float, str]]:
    best: dict[str, tuple[str, float, str]] = {}
    if not tsv.is_file():
        return best
    with tsv.open() as fh:
        header = None
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            if header is None:
                header = ln.rstrip().split("\t")
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            row = dict(zip(header, p))
            try:
                qv = float(row["q-value"])
            except ValueError:
                continue
            qid = row["Query_ID"]
            if qid not in best or qv < best[qid][1]:
                best[qid] = (row["Target_ID"], qv, row.get("Orientation", "+"))
    return best


def ensure_meme(tf_dir: Path) -> Path | None:
    homer = tf_dir / "homer" / "homerMotifs.all.motifs"
    if not homer.is_file():
        return None
    out = tf_dir / "tomtom" / "homer_denovo.meme"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file() or out.stat().st_mtime < homer.stat().st_mtime:
        subprocess.check_call(
            ["python", str(HOMER2MEME), "--homer", str(homer), "--out", str(out)]
        )
    return out


def run_tomtom(query: Path, db: Path, outdir: Path, *, optional: bool = False) -> Path | None:
    outdir.mkdir(parents=True, exist_ok=True)
    tsv = outdir / "tomtom.tsv"
    if tsv.is_file() and tsv.stat().st_mtime >= max(query.stat().st_mtime, db.stat().st_mtime):
        return tsv
    cmd = [
        TOMTOM, "-no-ssc", "-oc", str(outdir), "-verbosity", "1",
        "-min-overlap", "5", "-dist", "pearson", "-thresh", "0.1",
        str(query), str(db),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        if optional:
            print(f"  [warn] tomtom failed for {outdir.name} (optional)")
            return None
        raise
    return tsv


def parse_homer_rank_meta(tf_dir: Path) -> list[dict]:
    rows = []
    hr = tf_dir / "homer" / "homerResults"
    infos = []
    for info in hr.glob("motif*.info.html"):
        if "RV" in info.name or "similar" in info.name:
            continue
        m = re.search(r"motif(\d+)\.info\.html$", info.name)
        if m:
            infos.append((int(m.group(1)), info))
    for rank, info in sorted(infos):
        text = info.read_text(errors="replace")
        bg = re.search(r"Best Match.*?>([^<]+)<", text, re.S | re.I)
        sc = re.search(r"Match Score[:\s]*([0-9.]+)", text, re.I)
        if not sc:
            sc = re.search(r"score[=:\s]+([0-9.]+)", text, re.I)
        denovo_svg = hr / f"motif{rank}.logo.svg"
        sm = re.search(r"(<svg\b.*?</svg>)", text, re.S | re.I)
        matched_svg_data = minify_svg(sm.group(1)) if sm else ""
        denovo_data = minify_svg(denovo_svg.read_text()) if denovo_svg.is_file() else ""
        motif_file = hr / f"motif{rank}.motif"
        cons = ""
        if motif_file.is_file():
            first = motif_file.read_text().splitlines()[0]
            cons = first.lstrip(">").split("\t")[0].split(" ")[0]
        rows.append({
            "rank": rank,
            "homer_best": (bg.group(1).strip() if bg else ""),
            "homer_score": (sc.group(1) if sc else ""),
            "denovo_svg": denovo_data,
            "matched_svg": matched_svg_data,
            "cons": cons,
        })
    return rows


def process_tree(tree: Path, html_out: Path, label: str, *, workers: int | None = None) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    if workers is None:
        workers = int(os.environ.get("REBUILD_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    tfs = [t for t in UNKNOWN if (tree / t / "homer" / "homerResults.html").is_file()]
    print(f"[{label}] {len(tfs)} TFs (tomtom workers={workers})", flush=True)

    def one(tf: str) -> str:
        tf_dir = tree / tf
        meme = ensure_meme(tf_dir)
        if not meme:
            return f"skip {tf}: no motifs"
        # Force rematch against codebook_top1 (separate from any prior merged-DB run).
        cb_dir = tf_dir / "tomtom" / "homer_codebook_top1"
        art_dir = tf_dir / "tomtom" / "homer_artifacts"
        # bust cache if DB changed
        for d in (cb_dir, art_dir):
            tsv = d / "tomtom.tsv"
            if tsv.is_file() and MEME_DB.is_file() and tsv.stat().st_mtime < MEME_DB.stat().st_mtime:
                tsv.unlink()
        run_tomtom(meme, MEME_DB, cb_dir)
        if ART_DB.is_file():
            run_tomtom(meme, ART_DB, art_dir, optional=True)
        return f"ok {tf}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, tf): tf for tf in tfs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            msg = fut.result()
            if done % 10 == 0 or done == len(tfs):
                print(f"  tomtom {done}/{len(tfs)} last={msg}", flush=True)
    print(f"[{label}] tomtom done", flush=True)

    col_order_counter: Counter[str] = Counter()
    per_tf: dict[str, dict] = {}
    for tf in tfs:
        tf_dir = tree / tf
        meta = parse_homer_rank_meta(tf_dir)
        best = parse_tomtom_best(tf_dir / "tomtom" / "homer_codebook_top1" / "tomtom.tsv")
        arts = parse_tomtom_best(tf_dir / "tomtom" / "homer_artifacts" / "tomtom.tsv")
        by_rank = {}
        for qid, (tgt, qv, ori) in best.items():
            m = re.match(r"denovo(\d+)_", qid)
            if m:
                by_rank[int(m.group(1))] = (qid, tgt, qv, ori)
        art_rank = {}
        for qid, (tgt, qv, ori) in arts.items():
            m = re.match(r"denovo(\d+)_", qid)
            if m:
                art_rank[int(m.group(1))] = (tgt, qv)

        cells = {}
        for r in meta:
            rank = r["rank"]
            if rank not in by_rank:
                continue
            qid, tgt, qv, ori = by_rank[rank]
            src = "cdbk"
            col = short_name(tgt)
            is_art = rank in art_rank and art_rank[rank][1] <= 0.1
            hit = {
                "q": qv,
                "target": tgt,
                "src": src,
                "artifact": is_art,
                "denovo": r["denovo_svg"],
                "matched": r["matched_svg"],
                "homer_best": r["homer_best"],
                "score_label": f"TOMTOM q={qv:.2g} (Codebook)",
            }
            prev = cells.get(col)
            if prev is None or qv < prev["q"]:
                cells[col] = hit
        for col in cells:
            col_order_counter[col] += 1
        per_tf[tf] = cells
        print(f"  {tf}: {len(cells)} Codebook cols", flush=True)

    cols = [c for c, _ in col_order_counter.most_common()]
    print(f"[{label}] {len(cols)} unique Codebook matched motifs", flush=True)

    css = """
    body{font-family:system-ui,sans-serif;font-size:13px;margin:16px}
    table{border-collapse:collapse}
    th,td{border:1px solid #ddd;padding:8px;vertical-align:top}
    th{background:#f5f5f5;position:sticky;top:0;z-index:2}
    th.tf{position:sticky;left:0;z-index:3;background:#eee;text-align:left}
    td.ns{text-align:center;color:#999;min-width:280px}
    td.hit{min-width:320px}
    td.hit.art{outline:2px solid #c44}
    .lab{font-size:10px;color:#666}
    .n{font-weight:400;color:#666;font-size:10px;margin-top:2px}
    .src{font-size:10px;color:#356}
    img{height:52px;width:auto;display:block}
    """
    parts = [
        f"<html><head><meta charset=utf-8><title>{htmllib.escape(label)}</title>"
        f"<style>{css}</style></head><body>",
        f"<h1>{htmllib.escape(label)}</h1>",
        "<p>De novo HOMER motifs rematched with TOMTOM vs <b>Codebook top1</b> "
        "(Hughes et al. 2026; 204 curated TF motifs). Columns = Codebook TF; "
        "cell shows HOMER-ref logo (if available) + de novo logo + TOMTOM q. "
        "Red outline = also matches Codebook artifact set. NS = no Codebook hit "
        "at q≤0.1 for that column.</p>",
        "<div style='overflow:auto;max-height:85vh'><table>",
        "<thead><tr><th class='tf'>TF</th>",
        f"<th colspan='{len(cols)}'>Best Codebook match</th></tr>",
        "<tr class='names'><th class='tf'></th>",
    ]
    for c in cols:
        n = sum(1 for tf in tfs if c in per_tf.get(tf, {}))
        parts.append(f"<th>{htmllib.escape(c)}<div class='n'>{n} TFs</div></th>")
    parts.append("</tr></thead><tbody>")
    for tf in tfs:
        parts.append(f"<tr><th class='tf'>{tf}</th>")
        cells = per_tf.get(tf, {})
        for c in cols:
            hit = cells.get(c)
            if not hit:
                parts.append("<td class='ns'>NS</td>")
                continue
            art = " art" if hit["artifact"] else ""
            parts.append(f"<td class='hit{art}'>")
            parts.append("<div class='lab'>HOMER ref</div>")
            if hit["matched"]:
                parts.append(f"<img src='{data_uri(hit['matched'])}' alt=''/>")
            else:
                parts.append(
                    f"<div class='lab'>{htmllib.escape(hit['homer_best'] or hit['target'])}</div>"
                )
            parts.append("<div class='lab'>De novo</div>")
            if hit["denovo"]:
                parts.append(f"<img src='{data_uri(hit['denovo'])}' alt=''/>")
            parts.append(f"<div class='src'>{htmllib.escape(hit['score_label'])}</div>")
            if hit["artifact"]:
                parts.append("<div class='lab' style='color:#c44'>Codebook artifact</div>")
            parts.append("</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div></body></html>")
    html_out.write_text("".join(parts))
    print(f"[{label}] wrote {html_out} ({html_out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    process_tree(
        MA / "motif_pmat",
        MA / "motif_pmat" / "homer_denovo_all.html",
        "HOMER de novo (pmat peaks) × Codebook top1",
    )
    process_tree(
        MA / "motif_imputed",
        MA / "motif_imputed" / "homer_denovo_all.html",
        "HOMER de novo (imputed peaks) × Codebook top1",
    )


if __name__ == "__main__":
    main()
