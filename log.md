- change manuscripts to springer's two column format.

- revise the template if needed to fit the journal's requirements as follows:

```
Figure Placement and Size
Figures should be submitted within the body of the text. Only if the file size of the manuscript causes problems in uploading it, the large figures should be submitted separately from the text.
When preparing your figures, size figures to fit in the column width.
For large-sized journals the figures should be 84 mm (for double-column text areas), or 174 mm (for single-column text areas) wide and not higher than 234 mm.
For small-sized journals, the figures should be 119 mm wide and not higher than 195 mm.
```

## Changes

Added a `\geometry` override in `manuscript.tex` and `manuscript.tex` right after the packages:

| Dimension | Journal requirement | Now renders (verified from log) |
|---|---|---|
| Full-width figures | 174 mm | text width = **174.00 mm** |
| Single-column figures | 84 mm | column width = **83.99 mm** (6 mm gutter) |
| Max figure height | 234 mm | text height = **233.99 mm** |
| Paper | — | A4 (kept for submission) |

Since all figures use relative widths (`\linewidth`, `\columnwidth`, `\textwidth` fractions), they now automatically comply:

- `figure*` diagrams (Figs. 0, 8) → 174.0 mm wide, 35–74 mm high ✓
- Single-column figures (Figs. 6, 7) → 84.0 mm wide, 32–55 mm high ✓
- Multi-panel figures (Figs. 1–5) → sub-panels 38–78 mm ✓
- **No figure exceeds the 234 mm height limit** (tallest is ~78 mm)

## Extra fix

`Mel_Wave/normal.png` was lowercase on disk while the source references `Normal.png` — worked by luck on this filesystem, but breaks on case-sensitive systems. Renamed to `Normal.png`.

## Verification

Both manuscripts compile with **0 errors**, 17 pages (down from 19 with the taller text area), and only invisible 0.6 pt paragraph overfulls remain.


- Page limit
Manuscripts submitted as Original Articles should be between six (6) and ten (10) pages, including references.

Manuscripts submitted as Review Articles should not exceed fifteen (15) pages, including references.

>remove some redundant content if unnecessary to meet the page limit.

>replace `JPE_submission/manuscript.tex` with `paper/manuscript.tex`, also the bibliography file `JPE_submission/references.bib` with `paper/references.bib`. All relative files are located in the `JPE_submission` directory.