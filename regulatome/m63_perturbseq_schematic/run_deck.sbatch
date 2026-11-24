#!/bin/bash
#SBATCH -J m63deck
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m63_perturbseq_schematic/out/m63_deck_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m48_stage1_top10k/deck_work
export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
rm -f out.pptx preview.pptx; rm -rf render
$PY /work/users/d/y/dyy12/XuLab/regulatome/m63_perturbseq_schematic/m63_deck.py in.pptx out.pptx || exit 1
$PY -c "
from pptx import Presentation
p=Presentation('out.pptx'); lst=p.slides._sldIdLst
for i,el in enumerate(list(lst)):
    if i>2: p.part.drop_rel(el.rId); lst.remove(el)
p.save('preview.pptx')"
module load libreoffice/26.2.5
mkdir render; soffice --headless --convert-to pdf --outdir render preview.pptx >/dev/null 2>&1; pdftoppm -r 110 -png render/preview.pdf render/s; rm -f preview.pptx; ls render
