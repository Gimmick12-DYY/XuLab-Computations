#!/bin/bash
#SBATCH -J m57q005
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out/m57_deck_q005_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m48_stage1_top10k/deck_work
export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
rm -f out.pptx preview.pptx; rm -rf render
$PY /work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/m57_deck_q005.py in.pptx out.pptx || exit 1
$PY -c "
from pptx import Presentation
p=Presentation('out.pptx'); lst=p.slides._sldIdLst
for i,el in enumerate(list(lst)):
    if not any(sh.name=='FIG5S1C-table-cutoff' for sh in p.slides[i].shapes): pass
keep=[i for i,s in enumerate(p.slides) if any(sh.name=='FIG5S1C-table-cutoff' for sh in s.shapes)]
for i,el in enumerate(list(lst)):
    if i not in keep: p.part.drop_rel(el.rId); lst.remove(el)
p.save('preview.pptx')"
module load libreoffice/26.2.5
mkdir render; soffice --headless --convert-to pdf --outdir render preview.pptx >/dev/null 2>&1; pdftoppm -r 80 -png render/preview.pdf render/s; rm -f preview.pptx; ls render
