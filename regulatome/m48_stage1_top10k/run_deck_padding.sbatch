#!/bin/bash
#SBATCH -J m48dp
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m48_stage1_top10k/out/m48_deck_padding_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m48_stage1_top10k/deck_work
export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps
PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
rm -f out.pptx preview.pptx; rm -rf render
$PY ../m48_padding.py || exit 1
$PY ../m48_deck_padding.py in.pptx out.pptx || exit 1
$PY -c "
from pptx import Presentation
p=Presentation('out.pptx'); lst=p.slides._sldIdLst; keep=set()
for i,s in enumerate(p.slides):
    if any(sh.name in ('FIG5S1B-table-peakset','FIG5S1B-table-padding') for sh in s.shapes): keep.add(i)
for i,el in enumerate(list(lst)):
    if i not in keep: p.part.drop_rel(el.rId); lst.remove(el)
p.save('preview.pptx'); print('preview slides', sorted(keep))"
module load libreoffice/26.2.5
mkdir render; soffice --headless --convert-to pdf --outdir render preview.pptx >/dev/null 2>&1; pdftoppm -r 80 -png render/preview.pdf render/s; rm -f preview.pptx; ls render
