import json
from pathlib import Path

import pytest
from eval_helpers import make_question_rows, write_questions_file

from biomedrag import config
from biomedrag.eval import questions as Q
from biomedrag.eval.build_set import parse_handwritten

EVAL_DIR = Path(__file__).resolve().parents[1] / 'eval'


def test_split_depends_only_on_the_id_and_is_roughly_balanced():
    ids = [f's{i}' for i in range(1000)]
    splits = [Q.split_of(i) for i in ids]
    assert splits == [Q.split_of(i) for i in ids]
    assert set(splits) == {'dev', 'test'}
    assert 400 < splits.count('dev') < 600


def test_load_questions_filters_and_adds_the_split(tmp_path):
    path = tmp_path / 'q.jsonl'
    write_questions_file(path)
    everything = Q.load_questions(path)
    assert len(everything) == 28 and all(r['split'] in ('dev', 'test') for r in everything)
    dev = Q.load_questions(path, split='dev')
    assert dev and all(r['split'] == 'dev' for r in dev)
    assert {r['source'] for r in Q.load_questions(path, source='handwritten')} == {'handwritten'}
    assert all(r['answerable'] for r in Q.load_questions(path, answerable=True))
    assert not any(r['answerable'] for r in Q.load_questions(path, answerable=False))


def test_missing_question_file_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_set'):
        Q.load_questions(tmp_path / 'nope.jsonl')


def test_load_questions_uses_the_configured_path(tmp_path, monkeypatch):
    write_questions_file(tmp_path / 'q.jsonl')
    monkeypatch.setattr(config, 'QUESTIONS_PATH', tmp_path / 'q.jsonl')
    assert len(Q.load_questions()) == 28


def test_a_sound_set_has_no_problems():
    assert Q.validate(make_question_rows(), {'1', '2', '3', '5'}) == []


def good(**changes):
    row = {'id': 'x1', 'question': 'q?', 'answerable': True, 'gold_pmids': ['1'], 'source': 'synthetic', 'kind': None}
    return {**row, **changes}


@pytest.mark.parametrize('rows, pmids, expected', [
    ([good(), good()], None, 'duplicate id x1'),
    ([good(question='  ')], None, 'empty question'),
    ([good(source='magic')], None, 'source must be one of'),
    ([good(gold_pmids=[])], None, 'at least one gold PMID'),
    ([good(gold_pmids=['1', '99'])], {'1'}, 'PMID(s) 99 are not in the index'),
    ([good(answerable=False, gold_pmids=['1'], kind='near_miss', source='unanswerable')], None, 'must not have gold'),
    ([good(answerable=False, gold_pmids=[], kind='other', source='unanswerable')], None, 'need kind'),
])
def test_validation_names_each_kind_of_problem(rows, pmids, expected):
    assert any(expected in p for p in Q.validate(rows, pmids))


def test_summary_counts_every_source_and_split():
    text = Q.summary(make_question_rows())
    assert 'synthetic' in text and 'unanswerable' in text and text.splitlines()[-1].startswith('all')
    assert text.splitlines()[-1].split()[-1] == '28'


def test_corpus_pmids_reads_the_chunk_file(tmp_path, monkeypatch):
    f = tmp_path / 'chunks_abstract.jsonl'
    f.write_text('\n'.join(json.dumps({'pmid': p}) for p in ('7', '8', '8')) + '\n')
    monkeypatch.setattr(config, 'chunks_path', lambda strategy: f)
    assert Q.corpus_pmids() == {'7', '8'}


# ---- the files that ship with the project ----

def test_the_shipped_unanswerable_set_is_sound():
    rows = Q.read_jsonl(EVAL_DIR / 'unanswerable.jsonl')
    assert len(rows) == 40
    assert sum(r['kind'] == 'out_of_domain' for r in rows) == 20
    assert sum(r['kind'] == 'near_miss' for r in rows) == 20
    assert Q.validate(rows) == []


def test_the_handwritten_file_parses():
    """Works for the shipped drafts AND for your edited file: the number of questions is yours to choose."""
    rows, unlabelled = parse_handwritten(EVAL_DIR / 'handwritten.txt')
    assert rows
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(r['question'].endswith('?') for r in rows)
    assert all(r['gold_pmids'] for r in rows if r['id'] not in unlabelled)
    assert all(p.isdigit() for r in rows for p in r['gold_pmids'])
