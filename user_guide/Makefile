# Build the F Ring User Guide.
# The sty/ directory carries a few standard packages (booktabs, caption,
# enumitem, fancyvrb, titlesec, float, xcolor, pmboxdraw, newunicodechar,
# upquote) for TeX installations that lack them.  On a full TeX Live they
# are harmless; delete sty/ to use your system versions.

TEXINPUTS := ./sty//:$(TEXINPUTS)
export TEXINPUTS

PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

# The archived guide must be PDF/A (PDS4 Standards Reference).  pdflatex emits
# plain PDF 1.5, so main.pdf is produced by converting the pdflatex output with
# ocrmypdf, which embeds the sRGB output intent and the PDF/A identification
# that a bare Ghostscript run does not.  No OCR is performed: --skip-text
# leaves the existing text layer untouched, and --optimize 0 leaves the figures
# uncompressed so nothing is resampled.
#
# The hyperref pdfa option in preamble.tex is required for this to be lossless:
# PDF/A-1 forbids non-printing annotations, so without it every hyperlink in
# the document is silently discarded during conversion.
PDFA = ocrmypdf --output-type pdfa-1 --skip-text --optimize 0 --quiet

all: main.pdf

# main-raw.pdf is the direct pdflatex output; main.pdf is its PDF/A form.
main-raw.pdf: main.tex preamble.tex fontmacros.tex frontmatter-body.tex \
          $(wildcard sections/*.tex) $(wildcard figures/*.png)
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex
	mv main.pdf main-raw.pdf

main.pdf: main-raw.pdf
	$(PDFA) main-raw.pdf $@

.PHONY: all clean
clean:
	rm -f main.pdf main-raw.pdf *.aux *.log *.out *.toc *.lof *.lot sections/*.aux
