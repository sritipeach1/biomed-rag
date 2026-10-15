# How answers are produced (Step 4)

```
question
  -> hybrid search (BM25 + dense, fused with RRF) -> cross-encoder reranker -> top 5 papers
  -> [refusal layer 1] best reranker score below REFUSE_BELOW?  -> refuse, no model call
  -> language model writes an answer from the 5 passages, citing [PMID:...]
  -> [refusal layer 2] model replies INSUFFICIENT_EVIDENCE?     -> refuse
  -> citation check: every cited PMID must be one of the 5 passages
       invalid or missing -> one retry with feedback -> still invalid -> refuse
  -> answer + the list of papers, with the cited ones marked
```

## Why two refusal layers

A score threshold alone cannot catch every unanswerable question. On 15 hand-checked questions
(top reranker score of the best paper):

| Kind of question | Example | Top score |
|---|---|---|
| answerable | CAR-T, MGMT, TERT, resection ... | 5.7 to 8.9 |
| answerable but vaguely worded | "why do brain tumours come back after surgery and radiotherapy?" | 3.5 |
| off-topic | type 2 diabetes in older adults | 0.9 |
| off-topic | vaccines and heart failure | -3.4 |
| on-topic but not in the corpus | "cost of a glioma operation in a private hospital" | 3.5 |

The last two on-topic rows are almost the same score, so no threshold separates them. The score
catches clearly off-topic questions cheaply (no model call). Questions that sound on-topic but are not
actually answered by the papers are caught by the model's `INSUFFICIENT_EVIDENCE` reply.

`REFUSE_BELOW` defaults to 2.0 as a starting point. The reranker's scores are not probabilities and
depend on the model and library version, so the threshold is calibrated again in Step 5 on a dev
set that includes about 40 unanswerable questions.

## What the citation check does and does not prove

It proves that every cited paper was really retrieved for this question (no invented PMIDs, no
pointing at papers the model never saw). It does **not** prove that the paper supports the sentence.
That is "faithfulness" and is measured separately in Step 5 with a calibrated LLM judge.
It also reports citation coverage (the share of sentences with a citation) without enforcing it,
because models often leave a closing summary sentence uncited.

## Backends

| | Ollama | Groq |
|---|---|---|
| where it runs | your computer | hosted |
| model (default) | `qwen2.5:3b` | `openai/gpt-oss-120b` |
| use for | development, privacy | the public demo |

Set `LLM_BACKEND`, `OLLAMA_MODEL` and `GROQ_MODEL` in `.env`, or pass `--backend` on the command line.

The models a Groq key may use differ between accounts. If you get a 404, list them with the
`/openai/v1/models` endpoint and set `GROQ_MODEL` in `.env`.
