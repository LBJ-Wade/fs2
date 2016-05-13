PDF         = fs.pdf
MANUSCRIPTS = fs.tex
SPELL       = aspell --lang=en_GB -t -c

all: $(PDF)

fs.pdf: fs.tex fs.idx
	pdflatex $<
	@grep -e 'may have changed' $(basename $<).log && rm $@; $(MAKE) $@ || echo "done."

%.idx: %.tex
	makeindex $*

PHONY: check clean

check: $(MANUSCRIPTS)
	@echo "spell check"
	@for file in $(MANUSCRIPTS); do \
	    $(SPELL) $$file; echo $$file; \
	done

clean:
	rm -f $(PDF) *.bbl *.blg *.bak *.log *.aux
