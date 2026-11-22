# Complete English engineering report

This edition translates the complete Chinese v0.3 report: nine chapters, four appendices, references, all 24 stages, the narrative, tables, figure captions, and code/video indexes. Experimental values, units, dataset splits, source attribution, and limitations are retained. This is a full translated edition, not an abridged report or a new set of experiments.

- PDF: [build/main_en.pdf](build/main_en.pdf)
- LaTeX entry: [main_en.tex](main_en.tex)
- Chinese report: [../build/main_zh.pdf](../build/main_zh.pdf)
- Playable videos: [GitHub gallery](../../../docs/DEMOS.md)

Figures and videos are shared with the Chinese report under `../figures/` and `../videos/`. Embedded text in original screenshots and experiment frames is preserved as part of the original record; all captions and surrounding explanations are translated. Video filenames and code paths are unchanged. The English edition has its own pagination.

Compile from this directory with `./build.sh`, using XeLaTeX/latexmk or Tectonic. Text, headings, and code use Times New Roman; equations use Times-compatible mathematical fonts. Install Times New Roman locally before compiling. Font files are not redistributed with the repository. No model or GPU is needed for compilation or reading.

`python3 validate.py` checks source coverage, table cells, labels, citations, shared media and PDF metadata. It requires Python 3 and `pdfinfo` from Poppler. `VALIDATION.json`, `TRANSLATION_CHECK.json`, and `FONT_INFO.json` record the release checks.

The English PDF has 176 pages. Historical non-ASCII filenames are kept literally; those filename characters use a Chinese font fallback. All English text uses Times New Roman. `FONT_INFO.json` verifies embedded fonts; font binaries are not included.

## v0.3 update

The edition adds a chapter on engineering maintenance and offline acceptance, together with twelve sets of recorded frames, saved model inputs, simulator labels, and a historical hardware photograph (72 figures and seven unchanged videos in total). The original twenty-four experimental stages and technical findings are retained. The 127 offline checks are engineering verification, not new model experiments. Chinese and English captions share the same figures; the previous PDFs are preserved in `../versions/`.
