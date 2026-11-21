# Paper figures

Publication figures, each built by code from versioned source data. One folder per figure:

- `figure3/` — Figure 3 (see `figure3/README.md`: rebuild commands, panel -> script -> source data table)

`common/style.py` is shared by all figures: Arial with live text in PDF (TrueType, `pdf.fonttype = 42`) and SVG (`<text>`,
`svg.fonttype = "none"`, explicit font attributes for Adobe Illustrator), palette C colours, and one `save()` that writes
`<name>.pdf`, `<name>.svg` and a 300 dpi `<name>.png` preview. Built panels go to `<figure>/output/` (not tracked).
