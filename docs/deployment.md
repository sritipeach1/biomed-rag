# Running and deploying the demo

The demo is a small FastAPI service (`src/biomedrag/api/`) with one web page (`static/index.html`).

## Run it on your computer

```
uvicorn biomedrag.api.app:app --port 8000
```

Open http://localhost:8000. The service loads the index from `index/abstract` and uses the model backend
named in `.env` (`LLM_BACKEND`). `GET /health` reports `loading`, `ok` or `error`.

| setting (`.env`) | meaning |
|---|---|
| `LLM_BACKEND`, `GROQ_API_KEY`, `GROQ_MODEL` | which language model writes the answers |
| `REFUSE_BELOW` | reranker score below which a question is declined (5.0, chosen on the dev questions) |
| `PROMPT_VERSION` | which instructions the model gets (v3) |
| `INDEX_REPO`, `HF_TOKEN` | Hugging Face dataset to download the index from when it is missing locally |
| `ALLOWED_ORIGIN` | websites allowed to call the API from a browser (comma-separated) |
| `RATE_PER_MINUTE` | questions one visitor may ask per minute |
| `DAILY_ANSWER_LIMIT` | model calls per day for everyone together (protects a free API allowance) |

## Saved example answers

The example buttons on the page are answered from `static/demo_cache.json`, so they are instant, cost nothing
and keep working when the free model tier is busy. To refresh them after changing the prompt or threshold:

```
python -m biomedrag.api.demo_cache --backend groq
```

The questions are listed in `docs/demo_questions.txt`. Read every saved answer before committing the file.

## Docker

```
docker build -t biomed-rag .
docker run --rm -p 7860:7860 --env-file .env -v "${PWD}/index:/app/index" biomed-rag
```

Open http://localhost:7860. Without the `-v` mount, set `INDEX_REPO` and `HF_TOKEN` and the index is
downloaded on first start.

## Hugging Face Space

1. Upload the index folder to a **private dataset** repository (`hf upload <you>/<name> ./index/abstract . --repo-type dataset --private`).
2. In GitHub: Settings → Secrets and variables → Actions. Add a secret `HF_TOKEN` (a token with write access)
   and a variable `HF_SPACE` (like `your-hf-name/biomed-rag`).
3. Push to `main`. The workflow runs the tests, then uploads the app to the Space (creating it if needed).
4. In the Space settings add secrets and variables: `LLM_BACKEND=groq`, `GROQ_API_KEY`, `GROQ_MODEL`,
   `INDEX_REPO`, `HF_TOKEN` (read access to the dataset), `REFUSE_BELOW=5.0`, `PROMPT_VERSION=v3`,
   `ALLOWED_ORIGIN` (the portfolio website).

Free Spaces sleep when idle; the first request afterwards takes a minute or two.
