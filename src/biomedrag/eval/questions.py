"""The evaluation question set: how it is stored, split and checked.

One line per question in eval/questions.jsonl:

  {"id": "s31234567", "question": "...", "answerable": true,
   "gold_pmids": ["31234567"], "source": "synthetic", "kind": null}

source  synthetic     an LLM wrote it from one abstract; that abstract is the gold answer
        handwritten   you wrote it and found the relevant papers on PubMed yourself
        unanswerable  the abstracts cannot answer it, so the right behaviour is to refuse
kind    only for unanswerable questions:  out_of_domain  or  near_miss (sounds like glioma)

Dev/test split: every question is assigned by a fixed hash of its id. Tune everything on dev. Look at
test once, at the very end. A number measured on questions you tuned on is not a result.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from biomedrag import config

SOURCES = ('synthetic', 'handwritten', 'unanswerable')
KINDS = ('out_of_domain', 'near_miss')


def split_of(qid: str) -> str:
    """'dev' or 'test'. Depends only on the id, so adding questions never moves existing ones."""
    return 'dev' if int(hashlib.md5(qid.encode('utf-8')).hexdigest(), 16) % 2 == 0 else 'test'


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)


def load_questions(path: Path | None = None, split: str | None = None,
                   source: str | None = None, answerable: bool | None = None) -> list[dict]:
    """Questions from questions.jsonl, optionally only one split, source, or answerable flag."""
    path = path or config.QUESTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Build it with: python -m biomedrag.eval.build_set')
    rows = read_jsonl(path)
    for r in rows:
        r['split'] = split_of(r['id'])
    if split:
        rows = [r for r in rows if r['split'] == split]
    if source:
        rows = [r for r in rows if r['source'] == source]
    if answerable is not None:
        rows = [r for r in rows if r['answerable'] == answerable]
    return rows


def corpus_pmids(strategy: str = 'abstract') -> set[str]:
    """Every paper that is in the index (the gold answers must be among them)."""
    path = config.chunks_path(strategy)
    with open(path, encoding='utf-8') as f:
        return {json.loads(line)['pmid'] for line in f if line.strip()}


def validate(rows: list[dict], pmids_in_index: set[str] | None = None) -> list[str]:
    """A list of problems, empty if the set is sound."""
    problems = []
    ids = Counter(r['id'] for r in rows)
    problems += [f'duplicate id {i}' for i, c in ids.items() if c > 1]
    for r in rows:
        qid = r['id']
        if not r.get('question', '').strip():
            problems.append(f'{qid}: empty question')
        if r.get('source') not in SOURCES:
            problems.append(f'{qid}: source must be one of {SOURCES}, got {r.get("source")!r}')
        if r['answerable']:
            if not r.get('gold_pmids'):
                problems.append(f'{qid}: an answerable question needs at least one gold PMID')
            elif pmids_in_index is not None:
                missing = [p for p in r['gold_pmids'] if p not in pmids_in_index]
                if missing:
                    problems.append(f'{qid}: PMID(s) {", ".join(missing)} are not in the index, so '
                                    f'the system could never find them')
        else:
            if r.get('gold_pmids'):
                problems.append(f'{qid}: an unanswerable question must not have gold PMIDs')
            if r.get('kind') not in KINDS:
                problems.append(f'{qid}: unanswerable questions need kind in {KINDS}')
    return problems


def summary(rows: list[dict]) -> str:
    """A small table: how many questions per source and split."""
    counts = Counter((r['source'], split_of(r['id'])) for r in rows)
    lines = [f'{"source":<14}{"dev":>6}{"test":>6}{"total":>7}']
    for source in SOURCES:
        d, t = counts[(source, 'dev')], counts[(source, 'test')]
        lines.append(f'{source:<14}{d:>6}{t:>6}{d + t:>7}')
    d = sum(v for (_, sp), v in counts.items() if sp == 'dev')
    t = sum(v for (_, sp), v in counts.items() if sp == 'test')
    lines.append(f'{"all":<14}{d:>6}{t:>6}{d + t:>7}')
    return '\n'.join(lines)
