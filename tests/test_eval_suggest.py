import sys
from typing import ClassVar

from eval_helpers import real_retriever

from biomedrag.eval import suggest


def test_pooling_merges_modes_and_puts_papers_found_by_more_searches_first():
    pooled = suggest.pool(real_retriever(), 'IDH mutation glioma survival', per_mode=3)
    pmids = [e['chunk']['pmid'] for e in pooled]
    assert pmids[0] == '1' and len(pmids) == len(set(pmids))
    assert len(pooled[0]['modes']) == len(suggest.POOL_MODES)
    counts = [len(e['modes']) for e in pooled]
    assert counts == sorted(counts, reverse=True)


def test_render_lists_every_question_with_links_and_current_labels():
    rows = [{'id': 'h001', 'question': 'IDH mutation glioma survival', 'gold_pmids': ['1']},
            {'id': 'h002', 'question': 'MGMT temozolomide', 'gold_pmids': []}]
    text = suggest.render(rows, real_retriever())
    assert '## h001  IDH mutation glioma survival' in text and 'Currently labelled: 1' in text
    assert '(none yet)' in text and 'https://pubmed.ncbi.nlm.nih.gov/1/' in text
    assert 'found by: bm25, dense, hybrid_rerank' in text


def test_the_command_writes_the_candidate_file(tmp_path, monkeypatch):
    hand = tmp_path / 'h.txt'
    hand.write_text('h001 | IDH mutation glioma survival |\n')
    out = tmp_path / 'data' / 'cands.md'
    monkeypatch.setattr(suggest, 'HANDWRITTEN', hand)
    monkeypatch.setattr(suggest, 'OUT', out)
    monkeypatch.setattr(suggest.Retriever, 'load', classmethod(lambda cls, *a, **k: real_retriever()))
    monkeypatch.setattr(sys, 'argv', ['suggest'])
    suggest.main()
    assert 'PMID 1' in out.read_text(encoding='utf-8')


def test_a_paper_found_by_every_search_beats_papers_found_by_one_search_at_a_better_rank():
    def chunk(pmid):
        return {'pmid': pmid, 'title': pmid, 'year': '2024', 'text': pmid}

    class Stub:
        lists: ClassVar[dict] = {'bm25': ['A', 'B'], 'dense': ['C', 'B'], 'hybrid_rerank': ['D', 'B']}

        def retrieve(self, query, mode, k_final):
            return [(chunk(p), None) for p in self.lists[mode]]

    order = [e['chunk']['pmid'] for e in suggest.pool(Stub(), 'q')]
    assert order[0] == 'B' and set(order) == {'A', 'B', 'C', 'D'}
