TEX = pdflatex
BIB = bibtex
SRC = paper
TEXFLAGS = -interaction=nonstopmode -halt-on-error

all: $(SRC).pdf

$(SRC).pdf: $(SRC).tex references.bib
	$(TEX) $(TEXFLAGS) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(TEXFLAGS) $(SRC)
	$(TEX) $(TEXFLAGS) $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(SRC).pdf

.PHONY: all clean distclean
