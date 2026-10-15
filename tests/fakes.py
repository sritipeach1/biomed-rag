"""Small stand-ins for the real models, so tests run in a second and need no downloads."""
import re
import zlib

import numpy as np

WORD = re.compile(r'[a-z0-9]+')


class FakeEncoder:
    """Turns text into a normalised bag-of-words vector. Texts that share words get similar vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.calls: list[int] = []          # how many texts each encode() call received

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append(len(texts))
        out = np.zeros((len(texts), self.dim), dtype='float32')
        for row, text in enumerate(texts):
            for word in WORD.findall(text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FakeCrossEncoder:
    """Scores a (question, passage) pair by how many question words appear in the passage."""

    def predict(self, pairs, batch_size=32):
        scores = []
        for query, passage in pairs:
            q, p = set(WORD.findall(query.lower())), set(WORD.findall(passage.lower()))
            scores.append(float(len(q & p)))
        return np.array(scores)


def make_chunks() -> list[dict]:
    """Eight passages from six papers. Paper 1 has two passages, to test one-result-per-paper."""
    rows = [
        ('1', 'IDH mutation and survival', 'IDH mutation predicts better survival in glioma patients.'),
        ('1', 'IDH mutation and survival', 'IDH mutation status also guides glioma classification.'),
        ('2', 'MGMT and temozolomide', 'MGMT promoter methylation predicts response to temozolomide.'),
        ('3', 'Deep learning segmentation', 'Deep learning segments tumor volumes automatically on MRI.'),
        ('4', 'Diabetes in older adults', 'Metformin is first line treatment for type 2 diabetes in older adults.'),
        ('5', 'CAR-T cells', 'CAR-T cell therapy targets glioblastoma antigens in early trials.'),
        ('6', 'Surgery extent', 'Extent of resection improves overall survival in glioblastoma.'),
        ('6', 'Surgery extent', 'Gross total resection was achieved in most patients.'),
    ]
    chunks = []
    seen: dict[str, int] = {}
    for pmid, title, body in rows:
        idx = seen.get(pmid, 0)
        seen[pmid] = idx + 1
        chunks.append({'chunk_id': f'{pmid}_{idx}', 'pmid': pmid, 'title': title, 'year': '2024',
                       'text': f'{title}. {body}'})
    return chunks


class FakeLLM:
    """Replies with scripted answers, one per call, and remembers what it was asked."""

    name, model = 'fake', 'fake-1'

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []       # the messages of each call

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError('the model was called more often than the test expected')
        return self.replies.pop(0)


class StubRetriever:
    """Returns a fixed list of (passage, score) pairs, whatever the question."""

    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def retrieve(self, query, mode='hybrid_rerank', k_final=5, **kwargs):
        self.calls.append({'query': query, 'mode': mode, 'k_final': k_final})
        return self.results[:k_final]


def make_results(scores=(7.5, 6.0, 4.5)) -> list[tuple[dict, float]]:
    """Passages for papers 1, 2, 3 of `make_chunks`, with the given reranker scores."""
    chunks = {c['pmid']: c for c in make_chunks()}
    return [(chunks[p], s) for p, s in zip(('1', '2', '3'), scores, strict=False)]
