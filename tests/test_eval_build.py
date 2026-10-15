import json

import pytest

from biomedrag.eval import build_set as bs

SYN = {'id': 's1', 'question': 'Syn question?', 'answerable': True, 'gold_pmids': ['1'], 'source': 'synthetic', 'kind': None}
UNANS = {'id': 'u1', 'question': 'Unrelated?', 'answerable': False, 'gold_pmids': [], 'source': 'unanswerable',
         'kind': 'out_of_domain'}


def test_parse_handwritten_reads_ids_questions_and_pmids(tmp_path):
    f = tmp_path / 'h.txt'
    f.write_text('# comment\n\nh001 | What is IDH? | 111, 222\nh002 | Other thing? | 333;444\nh003 | Not yet? |\n', encoding='utf-8')
    rows, unlabelled = bs.parse_handwritten(f)
    assert [r['id'] for r in rows] == ['h001', 'h002', 'h003']
    assert rows[0]['gold_pmids'] == ['111', '222'] and rows[1]['gold_pmids'] == ['333', '444']
    assert rows[0]['source'] == 'handwritten' and rows[0]['answerable'] is True
    assert unlabelled == ['h003']


@pytest.mark.parametrize('line, message', [
    ('h001 | only two parts', 'expected'),
    ('h001 | question | 12ab', 'must be numbers'),
    (' | question | 1', 'expected'),
])
def test_parse_handwritten_explains_bad_lines(tmp_path, line, message):
    f = tmp_path / 'h.txt'
    f.write_text(line + '\n')
    with pytest.raises(ValueError, match=message):
        bs.parse_handwritten(f)


@pytest.fixture
def files(tmp_path, monkeypatch):
    syn, hand, unans = tmp_path / 'syn.jsonl', tmp_path / 'hand.txt', tmp_path / 'unans.jsonl'
    syn.write_text(json.dumps(SYN) + '\n')
    hand.write_text('h001 | Hand question? | 2\n')
    unans.write_text(json.dumps(UNANS) + '\n')
    monkeypatch.setattr(bs, 'SYNTHETIC', syn)
    monkeypatch.setattr(bs, 'HANDWRITTEN', hand)
    monkeypatch.setattr(bs, 'UNANSWERABLE', unans)
    return tmp_path


def test_build_merges_the_three_groups_and_writes_the_file(files):
    out = files / 'questions.jsonl'
    rows, problems = bs.build(check_index=False, out=out)
    assert problems == [] and [r['id'] for r in rows] == ['s1', 'h001', 'u1']
    assert len(out.read_text().splitlines()) == 3


def test_unlabelled_handwritten_questions_block_the_build(files):
    (files / 'hand.txt').write_text('h001 | Hand question? |\n')
    out = files / 'questions.jsonl'
    _, problems = bs.build(check_index=False, out=out)
    assert any('no PMID yet' in p for p in problems) and not out.exists()


def test_skip_handwritten_builds_without_them(files):
    (files / 'hand.txt').write_text('h001 | Hand question? |\n')
    out = files / 'questions.jsonl'
    rows, problems = bs.build(skip_handwritten=True, check_index=False, out=out)
    assert problems == [] and [r['id'] for r in rows] == ['s1', 'u1']


def test_missing_synthetic_file_is_reported(files):
    (files / 'syn.jsonl').unlink()
    _, problems = bs.build(skip_handwritten=True, check_index=False, out=files / 'q.jsonl')
    assert any('make_synthetic' in p for p in problems)


def test_gold_papers_must_be_in_the_index(files, monkeypatch):
    monkeypatch.setattr(bs, 'corpus_pmids', lambda: {'1'})          # PMID 2 (hand-written) is missing
    _, problems = bs.build(out=files / 'q.jsonl')
    assert any('h001' in p and 'not in the index' in p for p in problems)
