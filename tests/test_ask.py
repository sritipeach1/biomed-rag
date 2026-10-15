"""Runs the real retriever (with the small fake models) through the pipeline and the command line."""
import json
import sys

import pytest
from fakes import FakeCrossEncoder, FakeEncoder, FakeLLM, make_chunks

from biomedrag import ask
from biomedrag.generate.pipeline import RAGPipeline
from biomedrag.index.bm25 import BM25Index
from biomedrag.index.dense import DenseIndex
from biomedrag.retrieve.rerank import Reranker
from biomedrag.retrieve.retriever import Retriever


def real_retriever():
    chunks = make_chunks()
    texts = [c['text'] for c in chunks]
    dense = DenseIndex(model=FakeEncoder())
    dense.build(dense.embed(texts))
    return Retriever(chunks, BM25Index(texts), dense, Reranker(model=FakeCrossEncoder()))


def test_end_to_end_with_the_real_retriever():
    llm = FakeLLM(['IDH mutation predicts better survival in glioma [PMID:1].'])
    pipe = RAGPipeline(real_retriever(), llm, refuse_below=2.0)
    a = pipe.answer('IDH mutation glioma survival')
    assert a.status == 'answered'
    assert a.sources[0].pmid == '1' and a.sources[0].cited
    assert a.top_score >= 2.0


def test_end_to_end_off_topic_question_is_refused_before_the_model():
    llm = FakeLLM([])
    pipe = RAGPipeline(real_retriever(), llm, refuse_below=2.0)
    a = pipe.answer('football world cup winner')
    assert a.status in ('refused_low_score', 'refused_no_results')
    assert llm.calls == []


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Make the command line use the fake retriever and a scripted model."""
    llm = FakeLLM(['IDH helps [PMID:1].', 'MGMT matters [PMID:2].'])
    monkeypatch.setattr(RAGPipeline, 'load',
                        classmethod(lambda cls, backend=None, strategy='abstract', **kw:
                                    cls(real_retriever(), llm, **kw)))
    return tmp_path


def test_command_line_prints_answers_and_saves_a_file(fake_cli, monkeypatch, capsys):
    qfile = fake_cli / 'questions.txt'
    qfile.write_text('# a comment\nIDH mutation glioma survival\n\nMGMT temozolomide response\n',
                     encoding='utf-8')
    out = fake_cli / 'run.jsonl'
    monkeypatch.setattr(sys, 'argv', ['ask', '--file', str(qfile), '--save', str(out),
                                      '--refuse-below', '1'])
    ask.main()
    printed = capsys.readouterr().out
    assert 'IDH helps' in printed and 'MGMT matters' in printed
    assert 'status: answered' in printed and 'Papers retrieved' in printed
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert [r['question'] for r in rows] == ['IDH mutation glioma survival',
                                            'MGMT temozolomide response']
    assert all(r['status'] == 'answered' for r in rows)


def test_command_line_needs_a_question(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['ask'])
    with pytest.raises(SystemExit) as e:
        ask.main()
    assert e.value.code == 2


def test_command_line_shows_model_problems_as_a_plain_message(monkeypatch):
    from biomedrag.generate.llm import LLMError

    class Down:
        name, model = 'down', 'x'

        def complete(self, messages):
            raise LLMError('Ollama is not running')

    monkeypatch.setattr(RAGPipeline, 'load',
                        classmethod(lambda cls, backend=None, strategy='abstract', **kw:
                                    cls(real_retriever(), Down(), **kw)))
    monkeypatch.setattr(sys, 'argv', ['ask', 'IDH mutation glioma survival', '--refuse-below', '1'])
    with pytest.raises(SystemExit) as e:
        ask.main()
    assert 'Ollama is not running' in str(e.value.code)


def test_show_raw_prints_the_model_reply_even_for_refusals(fake_cli, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['ask', 'IDH mutation glioma survival', '--refuse-below', '1',
                                      '--show-raw'])
    ask.main()
    printed = capsys.readouterr().out
    assert 'raw model reply, attempt 1' in printed and 'IDH helps [PMID:1].' in printed
