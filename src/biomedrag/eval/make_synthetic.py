"""Have a language model write one question per sampled abstract (about 150 of them).

  python -m biomedrag.eval.make_synthetic --n 150 --backend groq

The abstract each question came from is its gold answer. Writes eval/synthetic.jsonl.
If it stops (rate limit, closed laptop), run the same command again: finished abstracts are
remembered in data/synthetic_cache.jsonl and skipped.

Known bias, to state in your README: a question written from an abstract tends to share words with it,
which flatters keyword search. That is why the hand-written set is reported separately.
Read at least 30 of the questions and delete bad ones from eval/synthetic.jsonl by hand.
The file is never overwritten once it exists, so your deletions are safe.
"""
import argparse
import json
import random
import re
import sys

from biomedrag import config
from biomedrag.eval.questions import read_jsonl, write_jsonl
from biomedrag.generate.llm import LLMError, get_llm
from biomedrag.ingest.chunk import load_sample_docs

OUT = config.EVAL / 'synthetic.jsonl'
CACHE = config.DATA / 'synthetic_cache.jsonl'

SYSTEM = 'You write evaluation questions for a biomedical literature search system.'
PROMPT = """Read this PubMed abstract and write ONE question that a researcher could ask and that this \
abstract answers.

Rules:
- One specific sentence, 8 to 30 words, ending with a question mark.
- It must make sense on its own. Never write "this study", "the authors", "the patients in this cohort" \
or anything that points at the abstract.
- Do not copy the title.
- Output only the question, nothing else.

Title: {title}

Abstract: {abstract}"""

POINTS_AT_ABSTRACT = re.compile(
    r'\b(this|the present|the current|our) (study|paper|trial|article|analysis|report|cohort|work)\b'
    r'|\bthe authors?\b|\bthese (patients|findings|results|data)\b', re.IGNORECASE)
WORD = re.compile(r'[a-z0-9]+')


def clean_question(raw: str) -> str:
    q = raw.strip().splitlines()[0].strip() if raw.strip() else ''
    q = re.sub(r'^(question\s*:\s*)', '', q, flags=re.IGNORECASE)
    return q.strip().strip('"').strip("'").strip()


def title_overlap(question: str, title: str) -> float:
    a, b = set(WORD.findall(question.lower())), set(WORD.findall(title.lower()))
    return len(a & b) / max(1, len(a))


def rejection_reason(question: str, title: str) -> str | None:
    """Why a generated question is not usable, or None if it is fine."""
    n = len(question.split())
    if n < 6 or n > 35:
        return f'length {n} words'
    if not question.endswith('?'):
        return 'does not end with a question mark'
    if POINTS_AT_ABSTRACT.search(question):
        return 'points at the abstract'
    if title_overlap(question, title) > 0.6:
        return 'copies the title'
    return None


def generate(llm, docs: list[dict], n: int, seed: int = config.SEED,
             cache_path=CACHE, max_attempts_factor: int = 4) -> list[dict]:
    """Return up to n accepted question rows, calling the model only for abstracts not yet in the cache."""
    cached = {r['pmid']: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    order = random.Random(seed).sample(range(len(docs)), min(len(docs), n * max_attempts_factor))
    accepted: list[dict] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as cache:
        for i in order:
            if len(accepted) >= n:
                break
            doc = docs[i]
            row = cached.get(doc['pmid'])
            if row is None:
                raw = llm.complete([{'role': 'system', 'content': SYSTEM},
                                    {'role': 'user', 'content': PROMPT.format(
                                        title=doc['title'], abstract=doc['abstract'])}])
                question = clean_question(raw)
                reason = rejection_reason(question, doc['title'])
                row = {'pmid': doc['pmid'], 'question': question, 'accepted': reason is None,
                       'reason': reason}
                cache.write(json.dumps(row, ensure_ascii=False) + '\n')
                cache.flush()
                print(f'{len(accepted) + row["accepted"]:>4}/{n}  '
                      f'{"ok " if row["accepted"] else "no "} {question[:100]}'
                      f'{"" if row["accepted"] else "   <- " + row["reason"]}')
            if row['accepted']:
                accepted.append({'id': f's{doc["pmid"]}', 'question': row['question'], 'answerable': True,
                                 'gold_pmids': [doc['pmid']], 'source': 'synthetic', 'kind': None})
    return accepted


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Generate synthetic evaluation questions.')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--backend', choices=['ollama', 'groq'], default=None)
    ap.add_argument('--force', action='store_true', help='overwrite an existing eval/synthetic.jsonl')
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        sys.exit(f'{OUT} already exists, so your edits are safe. Use --force to regenerate it.')
    llm = get_llm(args.backend)
    print(f'model: {llm.name} / {llm.model}')
    try:
        rows = generate(llm, list(load_sample_docs()), args.n)
    except LLMError as e:
        sys.exit(f'\nStopped: {e}\nRun the same command again to continue where it stopped.')
    write_jsonl(rows, OUT)
    print(f'\nwrote {len(rows)} questions to {OUT}')
    if len(rows) < args.n:
        print(f'only {len(rows)} of {args.n} passed the checks; run again with a different --n or seed')
    print('Now read at least 30 of them and delete the bad ones from that file.')


if __name__ == '__main__':
    main()
