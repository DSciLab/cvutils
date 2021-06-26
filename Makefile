REQUIREMENTS   := requirements.txt
PIP            := pip
PYTHON         := python


.PHONY: all dep install clean dist


all: dep install build


dist: clean
	$(PYTHON) setup.py sdist


build: dist


dep: $(REQUIREMENTS)
	$(PIP) install -r $<


install: dep
	$(PIP) install .


clean:
	-rm -rf .eggs .tox build MANIFEST dist
