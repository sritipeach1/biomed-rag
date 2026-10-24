.PHONY: fetch stats chunk index search test lint api demo-cache docker-build docker-run

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

api:
	uvicorn biomedrag.api.app:app --port 8000

demo-cache:
	python -m biomedrag.api.demo_cache --backend groq

docker-build:
	docker build -t biomed-rag .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env -v "$(CURDIR)/index:/app/index" biomed-rag
