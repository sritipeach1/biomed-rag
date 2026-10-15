import json

import pytest
from fakes import FakeCrossEncoder, FakeEncoder, make_chunks

from biomedrag.index.build import build_index
from biomedrag.retrieve.rerank import Reranker
from biomedrag.retrieve.retriever import Retriever


@pytest.fixture
def chunks_file(tmp_path):
    path = tmp_path / 'chunks_abstract.jsonl'
    path.write_text(''.join(json.dumps(c) + '\n' for c in make_chunks()), encoding='utf-8')
    return path


def build(tmp_path, chunks_file, **kwargs):
    return build_index('abstract', chunks_file=chunks_file, index_root=tmp_path / 'index',
                       model=FakeEncoder(), **kwargs)


def load(tmp_path):
    return Retriever.load('abstract', index_root=tmp_path / 'index', embed_model=FakeEncoder(),
                          reranker=Reranker(model=FakeCrossEncoder()))


def test_build_writes_all_files_and_a_working_retriever(tmp_path, chunks_file):
    out = build(tmp_path, chunks_file)
    for name in ('chunks.jsonl', 'bm25.pkl', 'dense.faiss', 'meta.json'):
        assert (out / name).exists()
    assert not (out / 'parts').exists()                # temporary saved blocks are cleaned up

    results = load(tmp_path).retrieve('temozolomide methylation', mode='hybrid_rerank')
    assert results[0][0]['pmid'] == '2'


def test_limit_builds_a_small_separate_index(tmp_path, chunks_file):
    out = build(tmp_path, chunks_file, limit=3)
    assert out.name == 'abstract_n3'
    assert len((out / 'chunks.jsonl').read_text().splitlines()) == 3


def test_skip_dense_builds_only_the_keyword_index(tmp_path, chunks_file):
    out = build(tmp_path, chunks_file, skip_dense=True)
    assert (out / 'bm25.pkl').exists()
    assert not (out / 'dense.faiss').exists()


def test_stale_saved_blocks_from_other_passages_are_discarded(tmp_path, chunks_file):
    parts = tmp_path / 'index' / 'abstract' / 'parts'
    parts.mkdir(parents=True)
    (parts / 'fingerprint.txt').write_text('left over from a different chunk file')
    (parts / 'emb_00000.npy').write_bytes(b'garbage')
    build(tmp_path, chunks_file)                       # must not crash or reuse the garbage
    assert load(tmp_path).dense.index.ntotal == len(make_chunks())


def test_loading_mismatched_index_files_is_caught(tmp_path, chunks_file):
    out = build(tmp_path, chunks_file)
    lines = (out / 'chunks.jsonl').read_text().splitlines()
    (out / 'chunks.jsonl').write_text('\n'.join(lines[:-1]) + '\n')      # lose one passage
    with pytest.raises(RuntimeError, match='do not match'):
        load(tmp_path)


def test_loading_a_missing_index_gives_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match='biomedrag.index.build'):
        Retriever.load('abstract', index_root=tmp_path)
