import csv
import json

import pytest

from biomedrag.eval import labels as lb


def recs(n_answers=12):
    out = []
    for i in range(n_answers):
        verdicts = ['supported', 'supported', 'partial' if i % 3 == 0 else 'supported', 'uncited']
        out.append({'id': f'q{i}', 'question': f'Question {i}?',
                    'sentences': [{'n': k, 'text': f'q{i} sentence {k}', 'pmids': ['1'] if v != 'uncited' else [],
                                   'verdict': v, 'reason': ''} for k, v in enumerate(verdicts, start=1)], 'raw': ''})
    return out


def test_pick_mixes_flagged_and_supported_and_never_picks_uncited():
    chosen = lb.pick(recs(), n=9)
    assert len(chosen) == 9
    kinds = [c['verdict'] for c in chosen]
    assert kinds.count('supported') > 0 and kinds.count('partial') == 3        # a third are flagged
    assert 'uncited' not in kinds
    assert len({(c['id'], c['n']) for c in chosen}) == 9                        # no sentence twice


def test_pick_spreads_over_answers_before_repeating_one():
    chosen = lb.pick(recs(10), n=8, flagged_share=0)
    assert len({c['id'] for c in chosen}) == 8


def test_pick_is_repeatable_and_copes_with_too_few_sentences():
    assert [c['text'] for c in lb.pick(recs(), 9)] == [c['text'] for c in lb.pick(recs(), 9)]
    assert len(lb.pick(recs(1), n=30)) == 3


def test_make_writes_a_blind_sheet_and_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, 'SHEET', tmp_path / 'sheet.csv')
    monkeypatch.setattr(lb, 'KEY', tmp_path / 'key.csv')
    rows_by_id = {f'q{i}': {'sources': [{'pmid': '1', 'text': f'Paper text for q{i}'}]} for i in range(12)}
    assert lb.make(recs(), rows_by_id, n=6) == 6
    sheet = (tmp_path / 'sheet.csv').read_text(encoding='utf-8-sig')
    assert 'partial' not in sheet and 'supported' not in sheet                   # the judge's verdict is hidden
    assert 'Paper text for q' in sheet
    with open(tmp_path / 'key.csv', encoding='utf-8') as f:
        key = list(csv.DictReader(f))
    assert len(key) == 6 and {k['judge_verdict'] for k in key} == {'supported', 'partial'}
    assert all(k['judge_verdict'] == ('partial' if int(k['answer_id'][1:]) % 3 == 0 and k['sentence'].endswith('sentence 3') else 'supported') for k in key)


def test_cohen_kappa():
    same = ['s', 'p', 'n', 's']
    assert lb.cohen_kappa(same, same) == 1.0
    assert lb.cohen_kappa(['s', 's', 'n', 'n'], ['s', 'n', 's', 'n']) == pytest.approx(0.0)
    assert lb.cohen_kappa(['s', 's'], ['s', 's']) == 1.0                          # everything the same: not a crash


def fill_sheet(path, letters):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    for row, letter in zip(rows[1:], letters, strict=False):
        row[4] = letter
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)


def test_read_labels_accepts_letters_or_words_and_skips_blank_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, 'SHEET', tmp_path / 'sheet.csv')
    monkeypatch.setattr(lb, 'KEY', tmp_path / 'key.csv')
    lb.make(recs(), {f'q{i}': {'sources': []} for i in range(12)}, n=4)
    fill_sheet(lb.SHEET, ['S', 'partial', 'n', ''])
    got = lb.read_labels(lb.SHEET)
    assert sorted(got.values()) == ['partial', 'supported', 'unsupported'] and len(got) == 3
    fill_sheet(lb.SHEET, ['x'])
    with pytest.raises(ValueError):
        lb.read_labels(lb.SHEET)


def test_score_reports_agreement_and_the_direction_of_disagreement(tmp_path):
    key = tmp_path / 'key.csv'
    key.write_text('item,answer_id,judge_verdict,sentence\n1,a,supported,x\n2,a,supported,x\n'
                   '3,a,partial,x\n4,a,unsupported,x\n', encoding='utf-8')
    labels = {1: 'supported', 2: 'unsupported', 3: 'supported', 4: 'unsupported'}
    text = '\n'.join(lb.score(labels, key))
    assert 'agreement on all three verdicts : 50%' in text
    assert 'agreement on supported vs not   : 50%' in text
    assert 'judge too lenient)  items [2]' in text and 'judge too strict)   items [3]' in text


def test_main_make_then_score_end_to_end(tmp_path, monkeypatch, capsys):
    from biomedrag import config
    from biomedrag.eval import answers as ans
    from biomedrag.eval import judge as jd
    for mod in (ans, jd):
        monkeypatch.setattr(mod, 'ANSWERS', tmp_path)
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'res')
    monkeypatch.setattr(lb, 'SHEET', tmp_path / 'sheet.csv')
    monkeypatch.setattr(lb, 'KEY', tmp_path / 'key.csv')
    ans.answers_path('dev', 'runA').write_text(
        ''.join(json.dumps({'id': f'q{i}', 'sources': [{'pmid': '1', 'text': 'T'}]}) + '\n' for i in range(12)),
        encoding='utf-8')
    jd.judge_path('dev', 'runA', jd.DEFAULT_JUDGE).write_text(
        ''.join(json.dumps(r) + '\n' for r in recs()), encoding='utf-8')
    assert lb.main(['make', '--run', 'runA', '--n', '6']) == 0
    with pytest.raises(SystemExit):
        lb.main(['make', '--run', 'runA', '--n', '6'])                            # will not overwrite your labels
    with pytest.raises(SystemExit):
        lb.main(['score'])                                                        # nothing filled in yet
    fill_sheet(lb.SHEET, ['s'] * 6)
    assert lb.main(['score']) == 0
    assert 'items you labelled: 6 of 6' in capsys.readouterr().out
    assert (tmp_path / 'res' / 'judge_agreement.txt').exists()


def test_controls_are_mixed_into_the_sheet_unmarked_and_remembered_in_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, 'SHEET', tmp_path / 'sheet.csv')
    monkeypatch.setattr(lb, 'KEY', tmp_path / 'key.csv')
    rows_by_id = {f'q{i}': {'sources': [{'pmid': '1', 'text': f'Paper text for q{i}'}]} for i in range(12)}
    controls = [{'id': 'c_q5', 'question': 'Q?', 'raw': '', 'sentences': [
        {'n': 1, 'text': 'A swapped sentence', 'pmids': ['1'], 'verdict': 'unsupported', 'reason': ''}]}]
    assert lb.make(recs(), rows_by_id, n=6, controls=controls) == 6
    sheet = (tmp_path / 'sheet.csv').read_text(encoding='utf-8-sig')
    assert 'A swapped sentence' in sheet and 'Paper text for q5' in sheet           # looks up the ORIGINAL answer's papers
    assert 'control' not in sheet.lower().replace('your_label', '')                  # nothing tells you it is one
    with open(tmp_path / 'key.csv', encoding='utf-8') as f:
        key = list(csv.DictReader(f))
    assert [k['control'] for k in key].count('yes') == 1 and len(key) == 6


def test_score_reports_how_the_controls_were_handled(tmp_path):
    key = tmp_path / 'key.csv'
    key.write_text('item,answer_id,judge_verdict,sentence,control\n1,a,supported,x,\n'
                   '2,c_a,supported,x,yes\n3,c_b,unsupported,x,yes\n', encoding='utf-8')
    text = '\n'.join(lb.score({1: 'supported', 2: 'unsupported', 3: 'unsupported'}, key))
    assert 'control sentences (hidden in your sheet): 2' in text
    assert 'you called 2 of 2 not fully supported; the judge flagged 1 of 2' in text
    old = tmp_path / 'old.csv'                                                       # keys made before controls existed
    old.write_text('item,answer_id,judge_verdict,sentence\n1,a,supported,x\n', encoding='utf-8')
    assert 'control' not in '\n'.join(lb.score({1: 'supported'}, old))
