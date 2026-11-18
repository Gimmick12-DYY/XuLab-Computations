#!/bin/bash
#SBATCH -J m45d
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary/out/m45_deck_bulk_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary/deck_work
export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps
PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
rm -f out3.pptx
$PY ../m45_deck_bulk.py in3.pptx out3.pptx || exit 1
$PY -c "
from pptx import Presentation
p=Presentation('out3.pptx'); lst=p.slides._sldIdLst
for i,el in enumerate(list(lst)):
    if i<12 or i>19: p.part.drop_rel(el.rId); lst.remove(el)
p.save('preview.pptx')"
module load libreoffice/26.2.5
rm -rf render; mkdir render; soffice --headless --convert-to pdf --outdir render preview.pptx >/dev/null 2>&1; pdftoppm -r 80 -png render/preview.pdf render/s; rm -f preview.pptx; ls render
