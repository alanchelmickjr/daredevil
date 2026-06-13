# Daredevil — dev convenience targets. Run `make` (or `make help`) for the list.
# $DAREDEVIL_HOME overrides where voiceprints/calibration live (default ~/.daredevil).
DAREDEVIL_HOME ?= $(HOME)/.daredevil

.PHONY: help install install-full live listen devices recalibrate test

help:
	@echo "Daredevil quickstart:"
	@echo "  make install       pip install -e .[audio]  (live mic + the 'daredevil' command)"
	@echo "  make install-full  add real backends (torch/ECAPA, DOA, prosody) — heavy"
	@echo "  make live          run the live web HUD  ->  http://127.0.0.1:8770"
	@echo "  make listen        one live awareness map -> stdout ('does it hear me?' check)"
	@echo "  make devices       show detected array + installed backends"
	@echo "  make recalibrate   clear the saved calibration, then run a fresh LIVE session"
	@echo "  make test          run the test suite"

install:
	pip install -e ".[audio]"

install-full:
	pip install -e ".[full]"

live:
	python -m daredevil serve --live

listen:
	python -m daredevil listen --live --json

devices:
	python -m daredevil devices

recalibrate:
	rm -f "$(DAREDEVIL_HOME)/calibration.json"
	python -m daredevil calibrate --live

test:
	python -m pytest -q
