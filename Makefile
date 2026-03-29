VENV     := venv
PYTHON   := $(VENV)/bin/python3
PIP      := $(VENV)/bin/pip
UVICORN  := $(VENV)/bin/uvicorn

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  VisionAI — available commands"
	@echo ""
	@echo "  make setup      create venv + install all dependencies"
	@echo "  make demo       run the demo (works before setup too)"
	@echo "  make test       run all 56 tests"
	@echo "  make serve      start REST API on localhost:8000"
	@echo "  make ui         open the browser UI"
	@echo "  make clean      remove venv, outputs, __pycache__"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: setup
setup: $(VENV)/bin/activate
	@echo ""
	@echo "  ✓  Ready. Run:  make demo"
	@echo ""

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: demo
demo:
	python3 demos/run_demo.py

.PHONY: test
test:
	python3 -m unittest discover -s tests -v

.PHONY: serve
serve: $(VENV)/bin/activate
	$(UVICORN) api.server:app --host 0.0.0.0 --port 8000 --reload

.PHONY: ui
ui:
	open web/index.html

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(VENV) outputs __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "  ✓  Cleaned"
