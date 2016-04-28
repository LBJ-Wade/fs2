SRC  := $(wildcard *.md)
HTML := $(addsuffix .html, $(basename $(SRC)))

all: $(HTML)

$(HTML): Makefile doc.css

%.html: %.md
	@echo $<
	pandoc --no-highlight --mathjax -c doc.css -c syntax.css $< -o $@

.PHONY: clean

clean:
	rm -f $(HTML)

