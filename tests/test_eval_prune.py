import json

import pytest

from biomedrag.eval import prune as pr


def make_file(tmp_path, n=5):
    p = tmp_path / 'synthetic.jsonl'
    p.write_text(''.join(json.dumps({'id': f's{i}', 'question': f'Q{i}?', 'answerable': True,
                                     'gold_pmids': [str(i)], 'source': 'synthetic', 'kind': None}) + '\n'
                         for i in range(n)), encoding='utf-8')
    return p


def ids_in(p):
    return [json.loads(line)['id'] for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_parse_ids_handles_spaces_commas_comments_and_duplicates():
    text = '# header\ns1 s2, s3  # trailing note\n\ns2\n  s4\n'
    assert pr.parse_ids(text) == ['s1', 's2', 's3', 's4']


def test_prune_removes_only_listed_ids_and_keeps_order(tmp_path):
    p = make_file(tmp_path)
    kept, removed, missing = pr.prune(['s1', 's3'], p, tmp_path / 'bak.jsonl')
    assert (kept, removed, missing) == (3, 2, [])
    assert ids_in(p) == ['s0', 's2', 's4']


def test_prune_saves_original_once(tmp_path):
    p, bak = make_file(tmp_path), tmp_path / 'bak.jsonl'
    pr.prune(['s1'], p, bak)
    pr.prune(['s2'], p, bak)
    assert ids_in(bak) == ['s0', 's1', 's2', 's3', 's4']   # still the untouched original


def test_prune_reports_unknown_ids_and_is_safe_to_rerun(tmp_path):
    p = make_file(tmp_path)
    pr.prune(['s1'], p, tmp_path / 'bak.jsonl')
    kept, removed, missing = pr.prune(['s1', 's99'], p, tmp_path / 'bak.jsonl')
    assert (kept, removed, missing) == (4, 0, ['s1', 's99'])
    assert ids_in(p) == ['s0', 's2', 's3', 's4']


def test_prune_with_no_matches_writes_no_backup(tmp_path):
    p, bak = make_file(tmp_path), tmp_path / 'bak.jsonl'
    pr.prune(['nope'], p, bak)
    assert not bak.exists()


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    p = make_file(tmp_path)
    monkeypatch.setattr(pr, 'TARGET', p)
    monkeypatch.setattr(pr, 'BACKUP', tmp_path / 'bak.jsonl')
    f = tmp_path / 'ids.txt'
    f.write_text('# c\ns0 s4\n', encoding='utf-8')
    assert pr.main(['--file', str(f)]) == 0
    assert ids_in(p) == ['s1', 's2', 's3']
    assert 'Removed 2, kept 3' in capsys.readouterr().out
    assert pr.main([]) == 1


@pytest.mark.parametrize('bad', ['', '# only a comment'])
def test_parse_ids_empty(bad):
    assert pr.parse_ids(bad) == []
