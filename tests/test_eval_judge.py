import json

import pytest
from fakes import FakeLLM

from biomedrag import config
from biomedrag.eval import answers as ans
from biomedrag.eval import judge as jd
from biomedrag.generate.llm import LLMError


def row(qid='q1', text='IDH predicts survival [PMID:1]. MGMT predicts response [PMID:2]. It is an overview.',
        status='answered'):
    return {'id': qid, 'question': 'Does IDH matter?', 'status': status, 'text': text,
            'sources': [{'pmid': '1', 'text': 'IDH mutation predicts better survival.'},
                        {'pmid': '2', 'text': 'MGMT methylation predicts response.'},
                        {'pmid': '3', 'text': 'Unrelated paper.'}]}


REPLY = json.dumps({'verdicts': [{'n': 1, 'verdict': 'supported', 'reason': 'states it'},
                                 {'n': 2, 'verdict': 'Partial', 'reason': 'no number'}]})


def test_claims_keep_their_citations_and_uncited_sentences_have_none():
    claims = jd.claims_of(row())
    assert [(c['n'], c['pmids']) for c in claims] == [(1, ['1']), (2, ['2']), (3, [])]


def test_the_judge_sees_only_cited_passages_and_cited_sentences():
    claims = jd.claims_of(row())
    user = jd.build_messages(claims, {s['pmid']: s['text'] for s in row()['sources']})[1]['content']
    assert '[PMID:1]\nIDH mutation' in user and '[PMID:2]\nMGMT' in user
    assert 'Unrelated paper' not in user                         # paper 3 was never cited
    assert 'It is an overview' not in user                       # the uncited sentence is not sent


@pytest.mark.parametrize('reply, expected', [
    (REPLY, {1: 'supported', 2: 'partial'}),                    # case is normalised
    ('```json\n' + REPLY + '\n```', {1: 'supported', 2: 'partial'}),
    ('<think>hmm {not json}</think>' + REPLY, {1: 'supported', 2: 'partial'}),
    ('Sure! ' + REPLY + ' Hope that helps.', {1: 'supported', 2: 'partial'}),
    ('not json at all', {}),
    ('{"verdicts": [{"n": 1, "verdict": "maybe"}, {"n": "x", "verdict": "supported"}, {"verdict": "supported"}]}', {}),
    ('{"verdicts": "oops"}', {}),
    ('[1, 2]', {}),
])
def test_parse_verdicts(reply, expected):
    assert {n: v['verdict'] for n, v in jd.parse_verdicts(reply).items()} == expected


def test_judge_answer_labels_every_sentence():
    llm = FakeLLM([REPLY])
    rec = jd.judge_answer(llm, row())
    assert [s['verdict'] for s in rec['sentences']] == ['supported', 'partial', 'uncited']
    assert len(llm.calls) == 1


def test_a_sentence_the_judge_forgot_is_unparsed_not_supported():
    rec = jd.judge_answer(FakeLLM([json.dumps({'verdicts': [{'n': 1, 'verdict': 'supported', 'reason': ''}]})]), row())
    assert [s['verdict'] for s in rec['sentences']] == ['supported', 'unparsed', 'uncited']


def test_an_answer_without_citations_costs_no_model_call():
    llm = FakeLLM([])
    rec = jd.judge_answer(llm, row(text='A statement without any citation.'))
    assert [s['verdict'] for s in rec['sentences']] == ['uncited'] and llm.calls == []


def test_only_answered_rows_are_judged_and_finished_ones_are_skipped(tmp_path):
    path = tmp_path / 'j.jsonl'
    rows = [row('q1'), row('q2', status='refused_by_model'), row('q3')]
    new, failed = jd.judge_run(FakeLLM([REPLY, REPLY]), rows, path, log=lambda *_: None)
    assert (new, failed) == (2, 0)
    llm = FakeLLM([REPLY])
    new, _ = jd.judge_run(llm, rows + [row('q4')], path, log=lambda *_: None)
    assert new == 1 and len(llm.calls) == 1
    assert set(ans.load_done(path)) == {'q1', 'q3', 'q4'}


def test_the_daily_allowance_stops_the_judge_at_once(tmp_path):
    class Daily:
        name, model = 'fake', 'm'

        def __init__(self):
            self.n = 0

        def complete(self, messages):
            self.n += 1
            raise LLMError('tokens per day (TPD) reached')
    llm = Daily()
    new, failed = jd.judge_run(llm, [row(f'q{i}') for i in range(5)], tmp_path / 'j.jsonl', log=lambda *_: None)
    assert (new, failed, llm.n) == (0, 1, 1)


def records(*verdict_lists):
    return [{'id': f'q{i}', 'question': 'Q?',
             'sentences': [{'n': n, 'text': f's{n}', 'pmids': ['1'] if v != 'uncited' else [], 'verdict': v, 'reason': ''}
                           for n, v in enumerate(vs, start=1)], 'raw': ''}
            for i, vs in enumerate(verdict_lists)]


def test_summary_counts_only_checked_sentences():
    s = jd.summarise(records(['supported', 'supported', 'uncited'], ['supported', 'partial', 'unsupported']))
    assert s['checked'] == 5 and s['sentences'] == 6 and len(s['uncited']) == 1
    assert s['rate']['supported'] == pytest.approx(3 / 5)
    assert s['per_answer_supported'] == [1.0, pytest.approx(1 / 3)]
    assert s['fully_supported_answers'] == [1.0, 0.0]


def test_the_report_lists_flagged_and_uncited_sentences():
    text = '\n'.join(jd.report(records(['supported', 'uncited'], ['partial']), 'run'))
    assert '[partial]' in text and 'sentences with no citation at all | 1 of 3' in text
    assert 'no checked sentences' in jd.report([], 'x')[2]


def test_summary_csv_has_one_row_per_run_and_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'RESULTS', tmp_path)
    recs = records(['supported', 'partial'])
    jd.save_summary(recs, 'dev', 'runA', 'm1')
    jd.save_summary(recs, 'dev', 'runA', 'm1')                  # again: replaces, does not duplicate
    jd.save_summary(recs, 'dev', 'runB', 'm1')
    lines = (tmp_path / 'faithfulness_dev.csv').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 3 and lines[1].startswith('runA,m1,1,2,0.5000')


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path)
    monkeypatch.setattr(jd, 'ANSWERS', tmp_path)
    monkeypatch.setattr(config, 'RESULTS', tmp_path / 'res')
    src = ans.answers_path('dev', 'runA')
    src.write_text(json.dumps(row('q1')) + '\n', encoding='utf-8')
    monkeypatch.setattr(jd, 'get_llm', lambda backend, model: FakeLLM([REPLY]))
    jd.main(['--split', 'dev', '--run', 'runA', '--sleep', '0'])
    out = capsys.readouterr().out
    assert '1 judged' in out and 'cited sentences supported | 50%' in out
    jd.main(['--split', 'dev', '--run', 'runA', '--report-only'])        # no model needed
    assert 'judged by' in capsys.readouterr().out


def test_the_test_split_needs_final(monkeypatch):
    with pytest.raises(SystemExit):
        jd.main(['--split', 'test', '--run', 'x'])


def answered(qid, text, scores=(7.0, 2.0, 5.0)):
    r = row(qid, text)
    for s, sc in zip(r['sources'], scores, strict=True):
        s['score'] = sc
    return r


def test_controls_swap_the_citation_and_keep_the_sentence():
    rows = [answered('q1', 'IDH predicts survival [PMID:1]. Another point [PMID:2].')]
    ctrl = jd.make_controls(rows, n=5)
    assert len(ctrl) == 1                                     # one control per answer
    c = ctrl[0]
    assert c['id'] == 'c_q1' and c['control_of'] == 'q1' and c['status'] == 'answered'
    assert c['text'] in ('IDH predicts survival [PMID:3].', 'IDH predicts survival [PMID:2].',
                         'Another point [PMID:1].', 'Another point [PMID:3].')


def test_controls_use_the_lowest_scoring_wrong_paper():
    r = answered('q1', 'IDH predicts survival [PMID:1].', scores=(7.0, 9.0, 1.0))
    (c,) = jd.make_controls([r], n=1)
    assert jd.extract_citations(c['text']) == ['3']           # paper 3 scored 1.0, the lowest of the uncited ones


def test_controls_skip_refusals_and_multi_cited_or_uncited_sentences_and_are_repeatable():
    rows = [answered('a', 'Fact [PMID:1][PMID:2]. No cite.'), row('b', status='refused_by_model'),
            answered('c', 'Fact [PMID:1].'), answered('d', 'Fact [PMID:2].')]
    first = jd.make_controls(rows, n=10)
    assert sorted(c['control_of'] for c in first) == ['c', 'd']
    assert [c['id'] for c in jd.make_controls(rows, n=10)] == [c['id'] for c in first]
    assert len(jd.make_controls(rows, n=1)) == 1


def test_a_judge_that_approves_wrong_paper_sentences_is_caught_by_the_control_report():
    lenient = records(['supported'], ['supported'], ['partial'])
    text = '\n'.join(jd.control_report(lenient))
    assert 'flagged 1 of 3 (33%)' in text and 'missed:' in text
    assert 'no control sentences' in jd.control_report([])[0]


def test_controls_end_to_end_with_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path)
    monkeypatch.setattr(jd, 'ANSWERS', tmp_path)
    src = ans.answers_path('dev', 'runA')
    src.write_text(json.dumps(answered('q1', 'IDH predicts survival [PMID:1].')) + '\n', encoding='utf-8')
    reply = json.dumps({'verdicts': [{'n': 1, 'verdict': 'unsupported', 'reason': 'wrong paper'}]})
    monkeypatch.setattr(jd, 'get_llm', lambda backend, model: FakeLLM([reply]))
    jd.main(['--split', 'dev', '--run', 'runA', '--controls', '3', '--sleep', '0'])
    out = capsys.readouterr().out
    assert 'flagged 1 of 1 (100%)' in out
    assert jd.controls_path('dev', 'runA', jd.DEFAULT_JUDGE).exists()
    assert not jd.judge_path('dev', 'runA', jd.DEFAULT_JUDGE).exists()       # the real verdicts are untouched
