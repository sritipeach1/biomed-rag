.PHONY: fetch stats test lint

fetch:
	python -m biomedrag.ingest.pubmed

stats:
	python -m biomedrag.ingest.stats

test:
	pytest -q

lint:
	ruff check src tests
