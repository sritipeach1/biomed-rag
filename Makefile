.PHONY: fetch stats chunk index search test lint

fetch:
	python -m biomedrag.ingest.pubmed

stats:
	python -m biomedrag.ingest.stats

chunk:
	python -m biomedrag.ingest.chunk

index:
	python -m biomedrag.index.build --strategy abstract

search:
	python -m biomedrag.search --file docs/manual_questions.txt --compare

test:
	pytest -q

lint:
	ruff check src tests
