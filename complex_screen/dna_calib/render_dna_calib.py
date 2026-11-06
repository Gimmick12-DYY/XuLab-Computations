# PyMOL render: crystal (grey) vs superposed Boltz-2 top model (blue), one PNG per target. Run with pymol -cq render_dna_calib.py -- <dir>
import sys
from pymol import cmd
D = sys.argv[1]
for t in ["8BYX_HOXB13_dimer_DNA", "1NKP_MYC_MAX_DNA"]:
    cmd.reinitialize()
    cmd.load(f"{D}/{t}_crystal_subset.pdb", "crystal"); cmd.load(f"{D}/{t}_top_superposed.pdb", "boltz")
    cmd.hide("everything"); cmd.show("cartoon")
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.color("grey70", "crystal"); cmd.color("tv_blue", "boltz and polymer.protein"); cmd.color("lightblue", "boltz and polymer.nucleic")
    cmd.set("cartoon_transparency", 0.15, "crystal")
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2)
    cmd.orient("crystal and polymer.nucleic"); cmd.turn("x", -90); cmd.zoom("all", 2)
    cmd.png(f"{D}/{t}_crystal_vs_boltz.png", width=1600, height=1200, dpi=300, ray=1)
    cmd.turn("x", 90)
    cmd.png(f"{D}/{t}_crystal_vs_boltz_side.png", width=1600, height=1200, dpi=300, ray=1)
