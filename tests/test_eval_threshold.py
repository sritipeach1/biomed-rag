import csv
import sys

import pytest
from eval_helpers import make_question_rows, real_retriever, write_questions_file

from biomedrag import config
from biomedrag.eval import threshold as T


def test_collect_scores_records_the_best_score_per_question():
    rows = T.collect_scores(real_retriever(), make_question_rows())
    assert len(rows) == 28
    idh = next(r for r in rows if r['id'] == 's00')
    assert idh['top_score'] >= 3 and idh['answerable'] is True
    off = next(r for r in rows if r['id'] == 'u00')
    assert off['answerable'] is False and (off['top_score'] is None or off['top_score'] < 1)


def test_rate_counts_answered_and_refused():
    rows = [{'top_score': 5.0}, {'top_score': 1.0}, {'top_score': None}]
    assert T.rate(rows, 2.0, True) == pytest.approx(1 / 3)
    assert T.rate(rows, 2.0, False) == pytest.approx(2 / 3)


def test_the_score_exactly_at_the_cutoff_is_answered():
    assert T.rate([{'top_score': 2.0}], 2.0, True) == 1.0


def test_breakdown_reports_each_group_separately():
    rows = [{'answerable': True, 'source': 'synthetic', 'kind': None, 'top_score': 5.0},
            {'answerable': True, 'source': 'handwritten', 'kind': None, 'top_score': 1.0},
            {'answerable': False, 'source': 'unanswerable', 'kind': 'out_of_domain', 'top_score': 0.0},
            {'answerable': False, 'source': 'unanswerable', 'kind': 'near_miss', 'top_score': 4.0}]
    text = '\n'.join(T.breakdown(rows, 2.0))
    assert 'synthetic' in text and 'handwritten' in text
    assert 'out_of_domain' in text and 'near_miss' in text
    near = next(ln for ln in text.splitlines() if 'near_miss' in ln)
    assert '0%' in near                                     # the near-miss question scored high: not refused


@pytest.fixture
def project(tmp_path, monkeypatch):
    write_questions_file(tmp_path / 'questions.jsonl')
    monkeypatch.setattr(config, 'QUESTIONS_PATH', tmp_path / 'questions.jsonl')
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(T.Retriever, 'load', classmethod(lambda cls, *a, **k: real_retriever()))
    return tmp_path


def test_the_test_split_needs_final_and_a_threshold_chosen_earlier(project, monkeypatch):
    for argv in (['--split', 'test'], ['--split', 'test', '--final'], ['--split', 'test', '--use', '2']):
        monkeypatch.setattr(sys, 'argv', ['threshold', *argv])
        with pytest.raises(SystemExit, match='reporting only'):
            T.main()


def test_a_dev_run_picks_a_threshold_and_writes_files(project, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    monkeypatch.setattr(sys, 'argv', ['threshold', '--split', 'dev'])
    T.main()
    out = capsys.readouterr().out
    assert 'refuse below' in out and 'answerable questions answered' in out and 'near_miss' in out
    sweep = list(csv.DictReader((project / 'results' / 'threshold_sweep_dev.csv').open(encoding='utf-8')))
    assert sweep and {'threshold', 'answered', 'refused'} == set(sweep[0])
    assert (project / 'results' / 'scores_dev.jsonl').exists()
    assert (project / 'docs' / 'threshold_dev.png').exists()


def test_the_final_run_uses_the_number_chosen_on_dev(project, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    monkeypatch.setattr(sys, 'argv', ['threshold', '--split', 'test', '--final', '--use', '2.0'])
    T.main()
    out = capsys.readouterr().out
    assert 'threshold 2.00 (given)' in out
