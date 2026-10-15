import csv
import json
import sys

import pytest
from eval_helpers import make_question_rows, real_retriever, write_questions_file

from biomedrag import config
from biomedrag.eval import retrieval as R


@pytest.fixture
def questions():
    return [q for q in make_question_rows() if q['answerable']]


def test_labels():
    assert R.label('bm25', None) == 'bm25' and R.label('hybrid_rerank', 20) == 'hybrid_rerank@20'


def test_run_config_records_where_the_gold_paper_landed(questions):
    rows = R.run_config(real_retriever(), questions, 'hybrid_rerank', 20)
    assert len(rows) == len(questions)
    idh = next(r for r in rows if r['id'] == 's00')
    assert idh['rank'] == 1 and idh['top_pmids'][0] == '1' and idh['ms'] >= 0


def test_a_missed_gold_paper_has_no_rank():
    q = [{'id': 'x', 'question': 'IDH mutation glioma survival', 'gold_pmids': ['4'], 'source': 'synthetic'}]
    rows = R.run_config(real_retriever(), q, 'bm25', None)
    assert rows[0]['rank'] is None or rows[0]['rank'] > 1


def test_summary_numbers_match_a_hand_calculation():
    rows = [{'rank': 1, 'ms': 10.0}, {'rank': 3, 'ms': 20.0}, {'rank': None, 'ms': 30.0}, {'rank': 8, 'ms': 40.0}]
    s = R.summarize(rows)
    assert s['n'] == 4 and s['recall@1'] == 0.25 and s['recall@5'] == 0.5 and s['recall@10'] == 0.75
    assert s['mrr@10'] == pytest.approx((1 + 1 / 3 + 0 + 1 / 8) / 4)
    assert s['p50_ms'] == 25.0 and s['recall@5_lo'] <= 0.5 <= s['recall@5_hi']


def test_rows_are_grouped_by_source(questions):
    rows = R.run_config(real_retriever(), questions, 'dense', None)
    groups = R.by_source(rows)
    assert set(groups) == {'synthetic', 'handwritten'} and len(groups['synthetic']) == 8


def test_csv_gets_one_header_and_keeps_appending(tmp_path):
    path = tmp_path / 'sub' / 'r.csv'
    rec = {c: 0.5 for c in R.CSV_COLUMNS}
    R.append_csv(path, [rec])
    R.append_csv(path, [rec, rec])
    lines = list(csv.reader(path.open(encoding='utf-8')))
    assert lines[0] == R.CSV_COLUMNS and len(lines) == 4


def test_markdown_table_has_a_row_per_system(questions):
    rows = R.run_config(real_retriever(), questions, 'bm25', None)
    text = R.markdown_table({'bm25': R.summarize(rows)}, 'synthetic questions')
    assert text.startswith('**synthetic questions**') and '| bm25 | 12 |' in text


def test_paired_comparison_reports_the_difference_and_whether_it_is_real():
    good = [{'id': str(i), 'source': 'synthetic', 'rank': 1} for i in range(30)]
    bad = [{'id': str(i), 'source': 'synthetic', 'rank': None} for i in range(30)]
    lines = R.paired_lines({'a': good, 'b': bad}, 'synthetic', 'a', 'b')
    assert any('recall@5: +1.000' in ln and 'a real difference' in ln for ln in lines)
    same = R.paired_lines({'a': good, 'b': good}, 'synthetic', 'a', 'b')
    assert any('could be chance' in ln for ln in same)
    assert R.paired_lines({'a': good, 'b': bad}, 'handwritten', 'a', 'b') == []


def test_failure_report_shows_gold_and_found_papers(capsys):
    results = [{'id': 'q1', 'rank': None, 'top_pmids': ['3', '4']}, {'id': 'q2', 'rank': 1, 'top_pmids': ['1']}]
    qs = {'q1': {'id': 'q1', 'question': 'Hard one?', 'gold_pmids': ['1']}, 'q2': {'id': 'q2', 'question': 'Easy?', 'gold_pmids': ['1']}}
    R.show_failures(results, qs, {'1': 'Gold title', '3': 'Wrong title'}, 5)
    out = capsys.readouterr().out
    assert 'Hard one?' in out and 'Gold title' in out and 'Wrong title' in out and 'Easy?' not in out


def test_the_chart_is_written(tmp_path, questions):
    pytest.importorskip('matplotlib')
    rows = R.run_config(real_retriever(), questions, 'bm25', None)
    summ = {'synthetic': {'bm25': R.summarize(rows), 'dense': R.summarize(rows)}}
    path = tmp_path / 'docs' / 'chart.png'
    R.plot(summ, path, 'title')
    assert path.stat().st_size > 2000


@pytest.fixture
def project(tmp_path, monkeypatch):
    write_questions_file(tmp_path / 'questions.jsonl')
    monkeypatch.setattr(config, 'QUESTIONS_PATH', tmp_path / 'questions.jsonl')
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(R.Retriever, 'load', classmethod(lambda cls, *a, **k: real_retriever()))
    return tmp_path


def test_the_test_split_is_locked_without_final(project, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['retrieval', '--split', 'test'])
    with pytest.raises(SystemExit, match='--final'):
        R.main()


def test_a_full_dev_run_writes_tables_files_and_a_chart(project, monkeypatch, capsys):
    pytest.importorskip('matplotlib')
    monkeypatch.setattr(sys, 'argv', ['retrieval', '--split', 'dev', '--failures', '2'])
    R.main()
    out = capsys.readouterr().out
    assert 'hybrid_rerank@20' in out and 'bm25' in out and 'recall@5' in out
    rows = list(csv.DictReader((project / 'results' / 'retrieval.csv').open(encoding='utf-8')))
    assert {r['mode'] for r in rows} == {'bm25', 'dense', 'hybrid', 'hybrid_rerank'}
    assert {r['split'] for r in rows} == {'dev'} and all(float(r['recall@5']) <= 1 for r in rows)
    perq = (project / 'results' / 'retrieval_abstract_dev_perquestion.jsonl').read_text().splitlines()
    assert json.loads(perq[0])['system'] == 'bm25'
    assert (project / 'docs' / 'retrieval_abstract_dev.png').exists()


def test_the_final_test_run_is_allowed_with_the_flag(project, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['retrieval', '--split', 'test', '--final', '--no-chart'])
    R.main()
    rows = list(csv.DictReader((project / 'results' / 'retrieval.csv').open(encoding='utf-8')))
    assert {r['split'] for r in rows} == {'test'}


def test_source_filter_limits_the_questions(project, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['retrieval', '--split', 'dev', '--source', 'handwritten', '--no-chart'])
    R.main()
    rows = list(csv.DictReader((project / 'results' / 'retrieval.csv').open(encoding='utf-8')))
    assert {r['source'] for r in rows} == {'handwritten'}
