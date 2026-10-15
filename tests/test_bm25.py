from fakes import make_chunks

from biomedrag.index.bm25 import BM25Index, tokenize


def texts():
    return [c['text'] for c in make_chunks()]


def test_tokenize_splits_hyphens_but_keeps_gene_names_whole():
    assert tokenize('IDH-wild-type IDH1 p53, TMZ.') == ['idh', 'wild', 'type', 'idh1', 'p53', 'tmz']


def test_best_match_ranks_first():
    index = BM25Index(texts())
    top = index.search('temozolomide methylation', k=3)
    assert top[0][0] == 2                              # the MGMT passage
    assert top[0][1] > 0


def test_results_are_sorted_best_first():
    scores = [s for _, s in BM25Index(texts()).search('glioma survival', k=8)]
    assert scores == sorted(scores, reverse=True)


def test_question_with_no_matching_word_returns_nothing():
    assert BM25Index(texts()).search('zebra quantum', k=5) == []


def test_asking_for_more_results_than_passages_is_fine():
    assert len(BM25Index(texts()).search('glioma', k=1000)) <= len(texts())


def test_save_and_load_give_the_same_results(tmp_path):
    index = BM25Index(texts())
    index.save(tmp_path / 'bm25.pkl')
    loaded = BM25Index.load(tmp_path / 'bm25.pkl')
    assert loaded.search('IDH mutation', k=3) == index.search('IDH mutation', k=3)
    assert loaded.size == len(texts())
