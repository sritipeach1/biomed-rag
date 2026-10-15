import numpy as np
from fakes import FakeEncoder, make_chunks

from biomedrag.index.dense import DenseIndex


def build_index(encoder=None):
    encoder = encoder or FakeEncoder()
    dense = DenseIndex(model=encoder)
    dense.build(dense.embed([c['text'] for c in make_chunks()]))
    return dense


def test_best_match_ranks_first():
    top = build_index().search('temozolomide response methylation', k=3)
    assert top[0][0] == 2


def test_scores_are_cosine_similarities_between_minus_one_and_one():
    scores = [s for _, s in build_index().search('glioma survival', k=8)]
    assert all(-1.001 <= s <= 1.001 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_asking_for_more_results_than_passages_is_fine():
    assert len(build_index().search('glioma', k=1000)) == len(make_chunks())


def test_save_and_load_give_the_same_results(tmp_path):
    dense = build_index()
    dense.save(tmp_path / 'dense.faiss')
    loaded = DenseIndex(model=FakeEncoder()).load(tmp_path / 'dense.faiss')
    assert loaded.search('IDH mutation', k=3) == dense.search('IDH mutation', k=3)


def test_resumable_embedding_only_redoes_missing_blocks(tmp_path):
    texts = [c['text'] for c in make_chunks()]            # 8 passages
    encoder = FakeEncoder()
    dense = DenseIndex(model=encoder)

    first = dense.embed_resumable(texts, tmp_path, block=3)       # blocks of 3, 3 and 2
    assert encoder.calls == [3, 3, 2]

    (tmp_path / 'emb_00001.npy').unlink()                         # pretend the run stopped early
    encoder.calls.clear()
    second = dense.embed_resumable(texts, tmp_path, block=3)
    assert encoder.calls == [3]                                   # only the lost block is redone
    assert np.allclose(first, second)
