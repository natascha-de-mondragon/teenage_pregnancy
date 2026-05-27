.PHONY: help install process test clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  process   Run both processing scripts to regenerate processed CSVs"
	@echo "  test      Run the test suite"
	@echo "  clean     Remove Python cache files"
	@echo ""
	@echo "Note: analysis steps live in notebooks/ and must be run manually in Jupyter."
	@echo "  notebooks/01_processing.ipynb  - exploratory processing"
	@echo "  notebooks/02_analysis.ipynb    - analysis and figures"

install:
	pip install -r requirements.txt

process:
	cd $(CURDIR) && python src/process_2018.py
	cd $(CURDIR) && python src/process_2023.py

test:
	pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
