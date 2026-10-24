import csv
import json
import sys

import pytest

from biomedrag import config
from biomedrag.eval import answer_report as rep
from biomedrag.eval import answers as ans


def row(qid, answerable, status='answered', score=7.0, gold=('1',), valid=('1',), n=4, cited=4, retries=0):
    report = {'cited': list(valid), 'valid': list(valid), 'invalid': [], 'n_sentences': n,
              'n_cited_sentences': cited} if status == 'answered' else None
    return {'id': qid, 'answerable': answerable, 'gold_pmids': list(gold) if answerable else [], 'source': 'x',
            'kind': None, 'status': status, 'top_score': score, 'retries': retries, 'report': report,
            'sources': [{'pmid': p, 'title': '', 'year': '', 'score': 1.0, 'cited': False, 'text': ''} for p in ('1', '2')],
            'retrieve_seconds': 0.5, 'llm_seconds': 1.5}


@pytest.mark.parametrize('status, score, threshold, expected', [
    ('answered', 7.0, None, 'answered'),
    ('answered', 7.0, 5.0, 'answered'),
    ('answered', 4.99, 5.0, 'refused_low_score'),
    ('answered', 5.0, 5.0, 'answered'),                       # equal passes, like the live pipeline
    ('refused_by_model', 9.0, 5.0, 'refused_by_model'),       # the threshold never turns a refusal into an answer
    ('answered', None, None, 'refused_no_results'),
    ('refused_no_results', None, 1.0, 'refused_no_results'),
])
def test_effective_status(status, score, threshold, expected):
    assert rep.effective_status(row('q', True, status, score), threshold) == expected


def test_decision_correct_for_both_kinds_of_question():
    assert rep.decision_correct(row('a', True), None) == 1.0
    assert rep.decision_correct(row('b', True, 'refused_by_model'), None) == 0.0
    assert rep.decision_correct(row('c', False, 'refused_by_model'), None) == 1.0
    assert rep.decision_correct(row('d', False), None) == 0.0


def test_a_higher_threshold_removes_leaks_but_also_refuses_good_questions():
    rows = [row('a1', True, score=8.0), row('a2', True, score=5.0), row('u1', False, score=6.0)]
    low = rep.evaluate(rows, 2.0)
    high = rep.evaluate(rows, 6.5)
    assert low['leaked'] == ['u1'] and low['wrongly_refused'] == {}
    assert high['leaked'] == [] and high['wrongly_refused'] == {'a2': 'refused_low_score'}
    assert low['answerable_answered'] == [1.0, 1.0] and high['answerable_answered'] == [1.0, 0.0]


def test_model_refusals_of_answerable_questions_are_counted_separately_from_score_refusals():
    rows = [row('a1', True, 'refused_by_model'), row('a2', True, 'refused_bad_citations'), row('a3', True, score=1.0)]
    r = rep.evaluate(rows, 3.0)
    assert r['refused_by_model_or_citations'] == 2
    assert r['wrongly_refused']['a3'] == 'refused_low_score'


def test_coverage_and_gold_cited_only_look_at_answered_questions():
    rows = [row('a1', True, n=4, cited=2, valid=('1',)),
            row('a2', True, n=2, cited=2, valid=('2',)),            # cites a paper, but not the gold one
            row('a3', True, 'refused_by_model')]
    r = rep.evaluate(rows, None)
    assert r['coverage'] == [0.5, 1.0]
    assert r['fully_cited'] == [0.0, 1.0]
    assert r['gold_cited'] == [1.0, 0.0]
    assert r['gold_retrieved'] == [1.0, 1.0, 1.0]                   # gold '1' is among the sources for all three


def test_latency_and_retries():
    rows = [row('a1', True, retries=1), row('a2', True)]
    r = rep.evaluate(rows, None)
    assert r['p50'] == pytest.approx(2.0)
    assert r['retried'] == [1.0, 0.0]


def test_the_comparison_names_the_questions_that_changed():
    base = [row('a1', True), row('u1', False)]                      # baseline leaks u1
    new = [row('a1', True), row('u1', False, 'refused_by_model')]
    text = '\n'.join(rep.compare_first_two({'v1': base, 'v2': new}, None))
    assert 'v2 minus v1' in text
    assert 'u1' in text and 'answered -> refused_by_model' in text
    assert 'a1' not in text
    assert rep.compare_first_two({'v1': base}, None) == []


def test_the_table_has_one_column_per_run():
    rows = [row('a1', True), row('u1', False)]
    lines = rep.table({'v1': rep.evaluate(rows, None), 'v2': rep.evaluate(rows, None)}, 5.0)
    assert lines[0] == '**threshold 5**'
    assert '| metric | v1 | v2 |' in lines


def test_end_to_end_report_writes_a_summary_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path / 'answers')
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'results')
    for tag, u1_status in (('runA', 'answered'), ('runB', 'refused_by_model')):
        path = ans.answers_path('dev', tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [row('a1', True), row('a2', True, score=3.0), row('u1', False, u1_status, score=6.0)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['r', '--split', 'dev', '--runs', 'runA', 'runB', '--thresholds', '5.0'])
    rep.main()
    out = capsys.readouterr().out
    assert 'threshold 5' in out and 'leaked unanswerable questions | 1: u1 | 0: -' in out
    with open(tmp_path / 'results' / 'answers_summary_dev.csv', encoding='utf-8') as f:
        got = list(csv.DictReader(f))
    leaked = {r['run']: r['value'] for r in got if r['metric'] == 'leaked unanswerable'}
    assert leaked == {'runA': '1', 'runB': '0'}


def test_runs_with_different_questions_are_compared_on_the_common_ones(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path / 'answers')
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'results')
    for tag, ids in (('runA', ['a1', 'a2']), ('runB', ['a1'])):
        path = ans.answers_path('dev', tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row(i, True)) + '\n' for i in ids), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['r', '--split', 'dev', '--runs', 'runA', 'runB', '--thresholds', '5'])
    rep.main()
    assert 'only the 1 that every run has' in capsys.readouterr().out


def test_the_test_split_needs_final(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['r', '--split', 'test', '--runs', 'x'])
    with pytest.raises(SystemExit):
        rep.main()
