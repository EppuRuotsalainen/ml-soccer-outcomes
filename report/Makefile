PDF=main

all: pdf

pdf:
	latexmk -pdf -shell-escape $(PDF).tex

clean:
	latexmk -C
	rm -f $(PDF).run.xml $(PDF).bcf

watch:
	latexmk -pvc -pdf -shell-escape $(PDF).tex
