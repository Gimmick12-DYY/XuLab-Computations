#!/bin/bash
#SBATCH -J m41pa -p general -t 6:00:00 --mem=32g -c 2 -o /work/users/d/y/dyy12/XuLab/regulatome/m41_partner_added_genes/out/m41_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m41_partner_added_genes
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python m41.py
