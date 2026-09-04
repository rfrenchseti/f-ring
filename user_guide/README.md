# Cassini ISS F Ring Mosaics User Guide — LaTeX source

Originally converted to LaTeX from `users-guide-draft-phase2-6.docx` (all
tracked changes accepted). These sources are now the master copy: several
sections have since been rewritten or added and have no Word counterpart.
Compiles to a 45-page PDF that follows the Word original's layout:
same section/figure numbering, US Letter geometry, portrait body with a
landscape Section 7, and "Page N of M" footers.

## Building

```
make            # runs pdflatex three times
```

or `pdflatex main.tex` (3×) with `TEXINPUTS=./sty//:` set. The `sty/`
directory carries standard packages (booktabs, caption, enumitem, fancyvrb,
fvextra, titlesec, float, xcolor, pmboxdraw, newunicodechar, upquote,
etoolbox, lastpage, lineno) for minimal TeX installations; on a full TeX Live
you can delete `sty/` and drop the `TEXINPUTS` override.

## Layout

| File | Contents |
| --- | --- |
| `main.tex` | Document driver: title block, TOC, section inputs |
| `preamble.tex` | Geometry, fonts, heading/caption/list/footer styling |
| `fontmacros.tex` | **Font substitution macros** (see below) and verbatim environments |
| `frontmatter-body.tex` | Title, version, author, DOI |
| `sections/01…07-*.tex` | One file per top-level section |
| `figures/*.png` | All 15 images extracted from the .docx, renamed descriptively |
| `citations/zotero-citations.json` | Full CSL JSON of every Zotero citation field (for the BibTeX pass) |
| `tools/convert-from-docx.py` | The converter script used to produce the sections (for reference/re-runs) |

## Font substitutions (acknowledged)

The Word original uses fonts that are not generally available to LaTeX.
Every special-font span is wrapped in a macro defined in `fontmacros.tex`,
so each can be re-pointed at the real font in one place (e.g. with
LuaLaTeX/XeLaTeX + `fontspec`):

| Word font | Role | Macro / environment | Current substitute |
| --- | --- | --- | --- |
| Cascadia Code | file names, program constants, inline code | `\code{…}` | Courier (`pcr`) |
| Cascadia Code | code blocks, directory trees, listings | `CodeBlock` env | Courier (`pcr`) |
| Courier New | Word "Literal" style | `\literal{…}`, `LiteralBlock` | Courier (`pcr`) |
| Lucida Sans | label listings in older drafts (unused after accepted changes) | `\labelxml{…}`, `LabelBlock` | Helvetica (`phv`) |
| Times New Roman | body text, title | document default | Times (`mathptmx`) |
| Calibri (bold) | headings | `\headingfont` | Helvetica (`phv`) |
| Minion | Word "Body Text" style (unused in final text) | — | — |

Other notes:

- Word wraps over-long code tokens and verbatim lines mid-token; this is
  reproduced with `\allowbreak` insertions and `fvextra`'s `breaklines`.
- The XML label listings use the smaller point size of the Word source
  (`[fontsize=\scriptsize]` on those blocks).
- On the landscape pages (Section 7) the footer appears rotated at the
  page edge — standard LaTeX `lscape`/`pdflscape` behavior.

## Citations / bibliography (second pass, pending)

The Word document manages citations with Zotero. For now each in-text
citation is wrapped as `\zcite{(Author Year)}` — a macro that just prints its
argument — and Section 6.5 (References) is plain formatted text. The full
CSL JSON for all 15 citation fields was extracted to
`citations/zotero-citations.json`. The planned second pass replaces `\zcite`
with `natbib`/BibTeX citations from the Zotero-exported `.bib` file and
regenerates the reference list.

## Known fidelity notes

- All tracked changes in the .docx were accepted; deleted figures/tables and
  their orphaned bookmarks were dropped. One cross-reference field near
  Figure 6 was already broken (empty) in Word and was omitted.
- Cross-reference targets: all 35 live `REF` fields resolve to the same
  section/figure numbers as in Word (verified against Word's cached field
  values).
