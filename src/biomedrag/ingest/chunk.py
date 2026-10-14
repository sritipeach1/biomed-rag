"""Clean the corpus, pick a random sample, and cut it into passages ("chunks").

Run:  python -m biomedrag.ingest.chunk

Reads   data/abstracts.jsonl        (Step 1)
Writes  data/docs_sample.jsonl      cleaned random sample of SAMPLE_SIZE abstracts
        data/chunks_abstract.jsonl  one passage per abstract
        data/chunks_sentences.jsonl overlapping 3-sentence windows
"""
import hashlib
import html
import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pysbd
from tqdm import tqdm

from biomedrag import config

# Only real formatting tags are removed, so "p < 0.05 and q > 0.01" is never damaged.
INLINE_TAG = re.compile(
    r'</?(?:i|b|u|sub|sup|em|strong|italic|bold|sc|span|math|mml:\w+)(?:\s[^>]*)?/?>', re.IGNORECASE)
BLOCK_TAG = re.compile(r'</?(?:p|br)(?:\s[^>]*)?/?>', re.IGNORECASE)
SPACES = re.compile(r'\s+')

# pysbd copes with "vs.", "i.e.", "et al.", "Dr." and "p < 0.05." but wrongly ends a sentence
# after these abbreviations when a number or a lowercase word follows ("ca. 15 months", "No. of").
PROTECTED_ABBREV = re.compile(r'\b(ca|approx|No|Nos|Fig|Figs|resp|incl)\.(?=\s+[a-z0-9])')
DOT_LEADER = '․'   # looks like a period but is not one, so the splitter ignores it


@lru_cache(maxsize=1)
def get_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language='en', clean=False)


def split_sentences(text: str) -> list[str]:
    protected = PROTECTED_ABBREV.sub(lambda m: m.group(1) + DOT_LEADER, text)
    sentences = (s.replace(DOT_LEADER, '.').strip() for s in get_segmenter().segment(protected))
    return [s for s in sentences if s]


def clean(text: str) -> str:
    text = INLINE_TAG.sub('', text)       # T<sub>1</sub> -> T1
    text = BLOCK_TAG.sub(' ', text)
    text = html.unescape(text)
    return SPACES.sub(' ', text).strip()


def _head(title: str) -> str:
    title = title.strip()
    if not title:
        return ''
    return title if title[-1] in '.?!' else title + '.'


def _row(doc: dict, idx: int, body: str) -> dict:
    return {
        'chunk_id': f"{doc['pmid']}_{idx}",
        'pmid': doc['pmid'],
        'title': doc['title'],
        'year': doc['year'],
        'text': f'{_head(doc["title"])} {body}'.strip(),   # the title goes in front of every passage
    }


def chunk_abstract(doc: dict) -> list[dict]:
    return [_row(doc, 0, doc['abstract'])]


def chunk_sentences(doc: dict, window: int = 3, stride: int = 2) -> list[dict]:
    sents = split_sentences(doc['abstract'])
    if not sents:
        return []
    rows = []
    starts = range(0, max(len(sents) - window + stride, 1), stride)
    for i, start in enumerate(starts):
        rows.append(_row(doc, i, ' '.join(sents[start:start + window])))
    return rows


STRATEGIES = {'abstract': chunk_abstract, 'sentences': chunk_sentences}


def load_clean_docs(path: Path = config.CORPUS_PATH, stats: Counter | None = None):
    """Yield cleaned documents. Counts what each filter removed into `stats`."""
    stats = stats if stats is not None else Counter()
    seen = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            stats['read'] += 1
            doc['title'] = clean(doc['title'])
            doc['abstract'] = clean(doc['abstract'])
            n_words = len(doc['abstract'].split())
            if n_words < config.MIN_WORDS:
                stats['too_short'] += 1
                continue
            if n_words > config.MAX_WORDS:
                stats['too_long'] += 1
                continue
            digest = hashlib.md5(doc['abstract'].encode('utf-8')).hexdigest()
            if digest in seen:                      # same text under another PMID
                stats['duplicate_text'] += 1
                continue
            seen.add(digest)
            stats['kept'] += 1
            yield doc


def sample_docs(docs, n: int = config.SAMPLE_SIZE, seed: int = config.SEED) -> list[dict]:
    """Random sample with a fixed seed, so every run picks the same abstracts."""
    docs = list(docs)
    if n >= len(docs):
        return docs
    picked = random.Random(seed).sample(docs, n)
    return sorted(picked, key=lambda d: int(d['pmid']))


def load_sample_docs(path: Path = config.SAMPLE_PATH):
    """The corpus the indexes are built from. Later steps (like question generation) must
    use this, never the full file, so every question's source paper is actually indexed."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows, total: int | None = None) -> int:
    n = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in tqdm(rows, total=total):
            f.write(json.dumps(row) + '\n')
            n += 1
    return n


def main():
    config.DATA.mkdir(exist_ok=True)

    stats = Counter()
    docs = list(load_clean_docs(stats=stats))
    print(f"read {stats['read']} abstracts")
    print(f"  removed {stats['too_short']} under {config.MIN_WORDS} words")
    print(f"  removed {stats['too_long']} over {config.MAX_WORDS} words")
    print(f"  removed {stats['duplicate_text']} with duplicate text")
    print(f"  kept    {stats['kept']}")

    sample = sample_docs(docs)
    write_jsonl(config.SAMPLE_PATH, sample)
    print(f'\nsample of {len(sample)} abstracts -> {config.SAMPLE_PATH}')

    for name, fn in STRATEGIES.items():
        out = config.chunks_path(name)
        print(f'\nchunking with "{name}" ...')
        rows = (row for doc in sample for row in fn(doc))
        n = write_jsonl(out, rows)
        print(f'{name}: {n} chunks ({n / len(sample):.1f} per abstract) -> {out}')


if __name__ == '__main__':
    main()
