import json
from pathlib import Path

from biomedrag import config
from biomedrag.index.bm25 import BM25Index
from biomedrag.index.dense import DenseIndex
from biomedrag.retrieve.hybrid import collapse_to_pmid, rrf
from biomedrag.retrieve.rerank import Reranker


class Retriever:
    """Finds the papers that best match a question. The four modes are the first ablation:

    bm25           keyword search only
    dense          meaning-based search only
    hybrid         both, merged with reciprocal rank fusion
    hybrid_rerank  hybrid, then a cross-encoder re-scores the best candidates
    """

    def __init__(self, chunks: list[dict], bm25: BM25Index, dense: DenseIndex,
                 reranker: Reranker | None = None):
        self.chunks, self.bm25, self.dense, self.reranker = chunks, bm25, dense, reranker

    def retrieve(self, query: str, mode: str = 'hybrid_rerank', k_final: int = config.K_FINAL,
                 k_cand: int = config.K_CANDIDATES,
                 k_rerank: int = config.K_RERANK) -> list[tuple[dict, float | None]]:
        """Return up to k_final (passage, score) pairs, one per paper, best first.
        The score is the reranker's score in 'hybrid_rerank' mode and None in the other modes."""
        if mode not in config.MODES:
            raise ValueError(f'unknown mode {mode!r}; choose one of {config.MODES}')
        if mode == 'hybrid_rerank' and self.reranker is None:
            raise ValueError('mode hybrid_rerank needs a reranker')

        # Run only the searches this mode needs, so timing each mode is fair.
        keyword = [i for i, _ in self.bm25.search(query, k_cand)] if mode != 'dense' else []
        meaning = [i for i, _ in self.dense.search(query, k_cand)] if mode != 'bm25' else []
        if mode == 'bm25':
            ids = keyword
        elif mode == 'dense':
            ids = meaning
        else:
            ids = rrf([keyword, meaning], top=k_cand)

        candidates = collapse_to_pmid([self.chunks[i] for i in ids])
        if mode == 'hybrid_rerank':
            return self.reranker.rerank(query, candidates[:k_rerank], top=k_final)
        return [(c, None) for c in candidates[:k_final]]

    @classmethod
    def load(cls, name: str, index_root: Path = config.INDEX, embed_model=None,
             reranker: Reranker | None = None, with_reranker: bool = True) -> 'Retriever':
        """Load an index folder written by `python -m biomedrag.index.build`.
        `embed_model` and `reranker` are only passed in by tests."""
        d = index_root / name
        for needed in ('chunks.jsonl', 'bm25.pkl', 'dense.faiss'):
            if not (d / needed).exists():
                raise FileNotFoundError(f'{d / needed} not found. Run: '
                                        f'python -m biomedrag.index.build --strategy {name}')
        with open(d / 'chunks.jsonl', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        meta = json.loads((d / 'meta.json').read_text(encoding='utf-8')) if (d / 'meta.json').exists() else {}
        bm25 = BM25Index.load(d / 'bm25.pkl')
        dense = DenseIndex(meta.get('embed_model', config.EMBED_MODEL), model=embed_model).load(d / 'dense.faiss')
        if not (len(chunks) == bm25.size == dense.index.ntotal):
            raise RuntimeError(f'index files in {d} do not match each other '
                               f'({len(chunks)} chunks, {bm25.size} keyword entries, '
                               f'{dense.index.ntotal} vectors). Rebuild the index.')
        if reranker is None and with_reranker:
            reranker = Reranker()
        return cls(chunks, bm25, dense, reranker)
