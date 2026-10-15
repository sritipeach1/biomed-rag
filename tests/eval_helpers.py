"""Shared set-up for the evaluation tests: a tiny real retriever (with fake models) and sample questions."""
import json

from fakes import FakeCrossEncoder, FakeEncoder, make_chunks

from biomedrag.index.bm25 import BM25Index
from biomedrag.index.dense import DenseIndex
from biomedrag.retrieve.rerank import Reranker
from biomedrag.retrieve.retriever import Retriever


def real_retriever() -> Retriever:
    chunks = make_chunks()
    texts = [c['text'] for c in chunks]
    dense = DenseIndex(model=FakeEncoder())
    dense.build(dense.embed(texts))
    return Retriever(chunks, BM25Index(texts), dense, Reranker(model=FakeCrossEncoder()))


ANSWERABLE = [
    ('IDH mutation glioma survival', ['1']),
    ('MGMT promoter methylation temozolomide response', ['2']),
    ('deep learning automatic tumor segmentation MRI', ['3']),
    ('CAR-T cell therapy glioblastoma antigens', ['5']),
]
UNANSWERABLE = [
    ('football world cup winner', 'out_of_domain'),
    ('cooking pasta carbonara recipe', 'out_of_domain'),
    ('hospital parking prices downtown', 'near_miss'),
    ('cinema ticket prices tonight', 'near_miss'),
]


def make_question_rows() -> list[dict]:
    """Enough questions that both the dev and the test split contain some of each group."""
    rows = []
    for i in range(12):
        q, gold = ANSWERABLE[i % len(ANSWERABLE)]
        source = 'synthetic' if i < 8 else 'handwritten'
        rows.append({'id': f'{source[0]}{i:02d}', 'question': q, 'answerable': True, 'gold_pmids': gold,
                     'source': source, 'kind': None})
    for i in range(16):
        q, kind = UNANSWERABLE[i % len(UNANSWERABLE)]
        rows.append({'id': f'u{i:02d}', 'question': q, 'answerable': False, 'gold_pmids': [],
                     'source': 'unanswerable', 'kind': kind})
    return rows


def write_questions_file(path, rows=None):
    rows = rows or make_question_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return rows
