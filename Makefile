.PHONY: fetch stats chunk test lint

fetch:
	python -m biomedrag.ingest.pubmed

stats:
	python -m biomedrag.ingest.stats

chunk:
	python -m biomedrag.ingest.chunk

test:
	pytest -q

lint:
	ruff check src tests
