# Evaluation set and results

| file | what it is | who writes it |
|---|---|---|
| `synthetic.jsonl` | about 150 questions an LLM wrote from sampled abstracts (gold = the source paper) | `make_synthetic`, then you delete bad ones |
| `handwritten.txt` | about 30 questions with the relevant PMIDs, found by you | you (`suggest` helps) |
| `unanswerable.jsonl` | 40 questions the abstracts cannot answer: 20 out of domain, 20 near-miss | provided, edit freely |
| `questions.jsonl` | the three groups merged and checked | `build_set` |
| `results/` | every measured table (CSV) and per-question output | the evaluation commands |

Questions are split 50/50 into **dev** and **test** by a hash of the question id. Tune on dev only; run
test once at the end with `--final`.

Commands, in order:

```
python -m biomedrag.eval.make_synthetic --n 150 --backend groq
python -m biomedrag.eval.suggest
python -m biomedrag.eval.build_set
python -m biomedrag.eval.retrieval --split dev
python -m biomedrag.eval.threshold --split dev
```
