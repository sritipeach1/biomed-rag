import pytest
from fakes import FakeCrossEncoder, FakeEncoder, make_chunks

from biomedrag import config
from biomedrag.index.bm25 import BM25Index
from biomedrag.index.dense import DenseIndex
from biomedrag.retrieve.rerank import Reranker
from biomedrag.retrieve.retriever import Retriever


def make_retriever(with_reranker=True):
    chunks = make_chunks()
    texts = [c['text'] for c in chunks]
    dense = DenseIndex(model=FakeEncoder())
    dense.build(dense.embed(texts))
    reranker = Reranker(model=FakeCrossEncoder()) if with_reranker else None
    return Retriever(chunks, BM25Index(texts), dense, reranker)


@pytest.mark.parametrize('mode', config.MODES)
def test_every_mode_finds_the_right_paper_with_one_result_per_paper(mode):
    results = make_retriever().retrieve('IDH mutation glioma survival', mode=mode, k_final=5)
    pmids = [chunk['pmid'] for chunk, _ in results]
    assert pmids[0] == '1'
    assert len(pmids) == len(set(pmids))               # paper 1 has two passages but appears once


def test_only_the_reranker_mode_returns_scores():
    r = make_retriever()
    assert all(s is None for _, s in r.retrieve('glioma', mode='hybrid'))
    scores = [s for _, s in r.retrieve('glioma survival', mode='hybrid_rerank')]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_k_final_limits_the_number_of_results():
    assert len(make_retriever().retrieve('glioma survival', mode='hybrid', k_final=2)) == 2


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='unknown mode'):
        make_retriever().retrieve('glioma', mode='magic')


def test_reranker_mode_without_a_reranker_is_rejected():
    with pytest.raises(ValueError, match='needs a reranker'):
        make_retriever(with_reranker=False).retrieve('glioma', mode='hybrid_rerank')


def test_each_mode_runs_only_the_searches_it_needs():
    class Forbidden:
        def search(self, *a, **k):
            raise AssertionError('this search should not have run')

    chunks = make_chunks()
    bm25 = BM25Index([c['text'] for c in chunks])
    dense = DenseIndex(model=FakeEncoder())
    dense.build(dense.embed([c['text'] for c in chunks]))

    Retriever(chunks, bm25, Forbidden()).retrieve('glioma', mode='bm25')       # no dense search
    Retriever(chunks, Forbidden(), dense).retrieve('glioma', mode='dense')     # no keyword search


def test_a_question_nothing_matches_gives_an_empty_list():
    assert make_retriever().retrieve('zebra quantum', mode='bm25') == []
