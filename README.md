# biomed-rag: a cited biomedical literature assistant

Ask a question about glioma (brain tumour) research. The system searches 30,000 PubMed abstracts and writes a
short answer in which **every sentence cites the paper it came from**. When the abstracts do not contain the
answer, it **declines instead of guessing**. A citation checker rejects answers that cite papers it never
retrieved, and an evaluation harness measures how often the system is right, how often it refuses correctly,
and whether the cited papers really support the sentences.

> Research tool only; not medical advice.

## Results on held-out test questions

95 questions (72 answerable, 23 unanswerable) that were not used to make any design choice. Run once.
Brackets are 95% bootstrap intervals. Full tables, failure analysis and limitations: [docs/results.md](docs/results.md).

| metric | result |
|---|---|
| answerable questions answered | 90% [83%, 97%] |
| unanswerable questions correctly refused | 100% (23 of 23) |
| right decision overall | 93% [87%, 98%] |
| right paper among the 5 retrieved | 97% [93%, 100%] |
| answers with a citation on every sentence | 95% [89%, 100%] |
| cited sentences supported by their paper (LLM judge) | 98% |
| time per question | 16.5 s typical, 21.8 s slowest 5% |

Things worth knowing before trusting these numbers:

- 95 questions is small. The intervals are wide, and 1 to 3 points between two settings is noise.
- Most questions are synthetic (written by an LLM from the abstract they test). These flatter keyword search.
  On the 8 hand-written test questions, reranking improves ranking (MRR@10 0.56 to 0.79); on the synthetic ones
  BM25 alone ranks best. See [section 2 of the results](docs/results.md#2-retrieval-which-search-method-finds-the-right-paper).
- The refusal threshold costs coverage: 6 of 72 answerable questions were refused because their score fell just
  under the cut.
- The faithfulness judge is itself a language model. It flagged 10 of 10 wrong-paper controls and agreed with
  hand labels on 28 of 30 sentences.

## How it works

```
question
  |  BM25 (keywords)  +  dense search (bge-small-en-v1.5)        top 50 each
  v  fused with reciprocal rank fusion (k = 60)
  |  cross-encoder reranker (ms-marco-MiniLM-L-6-v2) on the top 20
  v  best 5 papers (one passage per paper: title + abstract)
  |
  |-- refusal layer 1: best reranker score < 5.0 ?  --> decline, no model call
  v
  |  language model answers from the 5 passages, citing [PMID:12345678]
  |
  |-- refusal layer 2: model replies INSUFFICIENT_EVIDENCE ?  --> decline
  |-- citation check: every cited PMID must be one of the 5  -->  else one retry, then decline
  v
  answer + the 5 papers (cited ones marked), with links to PubMed
```

Design decisions and what the data said about them:

- **Hybrid search + rerank.** Keyword and dense search miss different questions. The reranker re-scores the top 20
  fused candidates; reranking more (50) was slower and never better.
- **Two refusal layers, because each leaks alone.** The score threshold misses look-alike questions that score
  high; the model's own refusal misses questions loosely related to a retrieved paper. Together they leaked 0 of
  40 unanswerable questions.
- **The threshold (5.0) was chosen on dev** from a plateau in the sweep, before the test split was run.
- **The prompt matters.** Requiring a citation on every sentence (prompt v3) raised "every sentence cited" from
  52% to 92% on dev without changing which questions were answered.
- **A small local model is not enough.** The same pipeline with a 3B model on a laptop refused almost every
  answerable question. The hosted `gpt-oss-120b` works, at about 15 s per answer.

## Run it

Needs Python 3.11 or newer. The commands are for PowerShell; on macOS or Linux, replace the activate line with
`source .venv/bin/activate`.

```powershell
git clone https://github.com/sritipeach1/biomed-rag
cd biomed-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # then edit .env
```

**1. Build the index** (one time; needs an NCBI email in `.env`; takes a while). The abstracts are not in this repo.

```powershell
python -m biomedrag.ingest.pubmed        # fetch abstracts (resumable)
python -m biomedrag.ingest.chunk         # clean, sample 30,000, write passages
python -m biomedrag.index.build --strategy abstract
```

**2. Pick a model backend** in `.env`:

- `LLM_BACKEND=groq`, plus `GROQ_API_KEY` (free tier available), uses `openai/gpt-oss-120b`.
- `LLM_BACKEND=ollama` uses a model on your own computer (`OLLAMA_MODEL`, default `qwen2.5:3b`).
  This runs without internet but refuses too often for real use.

**3. Ask a question**

```powershell
python -m biomedrag.ask "Does tumour treating fields therapy improve survival in newly diagnosed glioblastoma?"
python -m biomedrag.search "IDH mutation and survival" --compare     # retrieval only, all four modes
```

**4. Web demo and API**

```powershell
uvicorn biomedrag.api.app:app --port 8000     # open http://localhost:8000
```

`POST /ask`, `GET /health` and `GET /examples` are the endpoints. The service rate-limits visitors, caps model
calls per day, and answers a few saved example questions from a file. See [docs/deployment.md](docs/deployment.md)
for Docker and Hugging Face Spaces.

**5. Tests**

```powershell
python -m pytest -q
ruff check src tests
```

The tests use small fakes, so they need no model download, API key or network.

## Reproduce the evaluation

Questions live in `eval/` (`questions.jsonl`: 156 questions in dev and test). Tune on `dev`; `test` needs `--final`
and should be run once.

```powershell
# retrieval and refusal threshold
python -m biomedrag.eval.retrieval --split dev
python -m biomedrag.eval.threshold --split dev

# answers with a chosen prompt (resumable; Groq's free tier has a daily token limit)
python -m biomedrag.eval.answers --split dev --backend groq --prompt v3
python -m biomedrag.eval.answer_report --split dev --runs groq-openai-gpt-oss-120b-v2 groq-openai-gpt-oss-120b-v3

# faithfulness judge, wrong-paper controls, and agreement with hand labels
python -m biomedrag.eval.judge --split dev --run groq-openai-gpt-oss-120b-v3 --controls 10
python -m biomedrag.eval.labels make --run groq-openai-gpt-oss-120b-v3
python -m biomedrag.eval.labels score

# the final, one-time test run
python -m biomedrag.eval.answers --split test --final --backend groq --prompt v3
python -m biomedrag.eval.answer_report --split test --final --runs groq-openai-gpt-oss-120b-v3 --thresholds 5.0
python -m biomedrag.eval.retrieval --split test --final
python -m biomedrag.eval.threshold --split test --final --use 5.0
python -m biomedrag.eval.judge --split test --final --run groq-openai-gpt-oss-120b-v3
```

Result tables are written to `eval/results/` and are committed.

## Repository layout

```
src/biomedrag/
  ingest/      fetch from PubMed, clean, chunk
  index/       BM25 and dense (FAISS) indexes
  retrieve/    hybrid fusion, reranker, Retriever
  generate/    prompts, model clients, citation checker, RAGPipeline
  eval/        question set, retrieval/answer metrics, threshold sweep, faithfulness judge
  api/         FastAPI service, limits, saved examples
static/        the demo page
eval/          questions and result tables
docs/          results, answering design, deployment
tests/         pytest, using fakes
```

## Data and licence

Abstracts come from PubMed through the NCBI E-utilities. The full abstract set is not redistributed here and the
demo shows only titles and links to PubMed, not abstract text. Code is released under the MIT licence.
