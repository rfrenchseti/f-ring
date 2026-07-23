# Build the F Ring User's Guide.
# The sty/ directory carries a few standard packages (booktabs, caption,
# enumitem, fancyvrb, titlesec, float, xcolor, pmboxdraw, newunicodechar,
# upquote) for TeX installations that lack them.  On a full TeX Live they
# are harmless; delete sty/ to use your system versions.

TEXINPUTS := ./sty//:$(TEXINPUTS)
export TEXINPUTS

PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

main.pdf: main.tex preamble.tex fontmacros.tex frontmatter-body.tex \
          $(wildcard sections/*.tex) $(wildcard figures/*.png)
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex

.PHONY: clean
clean:
	rm -f main.pdf *.aux *.log *.out *.toc *.lof *.lot sections/*.aux
