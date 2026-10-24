import json
import sys

import pytest
from fakes import FakeLLM, StubRetriever, make_results

from biomedrag.eval import answers as ans
from biomedrag.generate.llm import LLMError
from biomedrag.generate.pipeline import RAGPipeline

GOOD = 'IDH mutation predicts better survival [PMID:1]. MGMT predicts response [PMID:2].'


def questions(n=3):
    return [{'id': f'q{i}', 'question': f'Question {i}?', 'answerable': i % 2 == 0, 'gold_pmids': ['1'] if i % 2 == 0 else [],
             'source': 'synthetic' if i % 2 == 0 else 'unanswerable', 'kind': None if i % 2 == 0 else 'near_miss'}
            for i in range(n)]


def pipe(replies, scores=(7.5, 6.0, 4.5), **kw):
    llm = FakeLLM(replies)
    return RAGPipeline(StubRetriever(make_results(scores)), llm, refuse_below=float('-inf'), **kw), llm


def saved(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_run_tag_is_safe_for_file_names():
    class L:
        name, model = 'groq', 'openai/gpt-oss-120b'
    assert ans.run_tag(L, 'v2') == 'groq-openai-gpt-oss-120b-v2'
    L.model = 'qwen2.5:3b'
    assert ans.run_tag(L, 'v1') == 'groq-qwen2.5-3b-v1'


def test_every_question_is_saved_with_its_labels_and_the_answer(tmp_path):
    p, _ = pipe([GOOD] * 3)
    path = tmp_path / 'a.jsonl'
    new, skipped, failed = ans.run(p, questions(), path, log=lambda *_: None)
    assert (new, skipped, failed) == (3, 0, 0)
    rows = saved(path)
    assert [r['id'] for r in rows] == ['q0', 'q1', 'q2']
    r = rows[0]
    assert r['answerable'] is True and r['gold_pmids'] == ['1'] and r['source'] == 'synthetic'
    assert r['status'] == 'answered' and r['report']['valid'] == ['1', '2']
    assert r['sources'][0]['text'].startswith('IDH mutation')           # passages kept for the judge


def test_the_score_threshold_is_not_applied_while_collecting(tmp_path):
    p, llm = pipe([GOOD], scores=(0.1, 0.05, 0.01))                     # hopeless scores
    ans.run(p, questions(1), tmp_path / 'a.jsonl', log=lambda *_: None)
    assert len(llm.calls) == 1                                          # the model was still asked
    assert saved(tmp_path / 'a.jsonl')[0]['top_score'] == 0.1


def test_finished_questions_are_skipped_on_the_next_run(tmp_path):
    path = tmp_path / 'a.jsonl'
    ans.run(pipe([GOOD] * 2)[0], questions(2), path, log=lambda *_: None)
    p, llm = pipe([GOOD])                                               # only ONE reply available
    new, skipped, _ = ans.run(p, questions(3), path, log=lambda *_: None)
    assert (new, skipped) == (1, 2)
    assert len(llm.calls) == 1
    assert [r['id'] for r in saved(path)] == ['q0', 'q1', 'q2']


def test_a_half_written_last_line_is_ignored_and_redone(tmp_path):
    path = tmp_path / 'a.jsonl'
    ans.run(pipe([GOOD])[0], questions(1), path, log=lambda *_: None)
    with open(path, 'a', encoding='utf-8') as f:
        f.write('{"id": "q1", "status": "ans')                          # crash in the middle of a write
    assert set(ans.load_done(path)) == {'q0'}


class FailingLLM:
    name, model = 'fake', 'fake-1'

    def __init__(self, fail_ids=()):
        self.n = 0

    def complete(self, messages):
        self.n += 1
        raise LLMError('rate limit')


def test_model_errors_are_not_saved_and_stop_the_run_after_three_in_a_row(tmp_path):
    p = RAGPipeline(StubRetriever(make_results()), FailingLLM(), refuse_below=float('-inf'), max_retries=0)
    path = tmp_path / 'a.jsonl'
    new, _, failed = ans.run(p, questions(6), path, log=lambda *_: None)
    assert (new, failed) == (0, 3)                                      # stopped early, not 6 failures
    assert not path.read_text(encoding='utf-8')


def test_an_error_between_successes_does_not_stop_the_run(tmp_path):
    class Flaky:
        name, model = 'fake', 'fake-1'

        def __init__(self):
            self.n = 0

        def complete(self, messages):
            self.n += 1
            if self.n == 2:
                raise LLMError('hiccup')
            return GOOD
    p = RAGPipeline(StubRetriever(make_results()), Flaky(), refuse_below=float('-inf'), max_retries=0)
    path = tmp_path / 'a.jsonl'
    new, _, failed = ans.run(p, questions(3), path, log=lambda *_: None)
    assert (new, failed) == (2, 1)
    assert [r['id'] for r in saved(path)] == ['q0', 'q2']


def test_the_prompt_version_reaches_the_model():
    p, llm = pipe([GOOD], prompt_version='v2')
    p.answer('q')
    assert 'Step 1' in llm.calls[0][0]['content']
    p1, llm1 = pipe([GOOD])
    p1.answer('q')
    assert 'Step 1' not in llm1.calls[0][0]['content']


def test_the_test_split_needs_final(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['answers', '--split', 'test', '--prompt', 'v1'])
    with pytest.raises(SystemExit) as e:
        ans.main()
    assert 'final' in str(e.value)


def test_main_switches_the_threshold_off_and_passes_the_prompt_version(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_load(backend, strategy, **kwargs):
        seen.update(kwargs)
        return RAGPipeline(StubRetriever(make_results()), FakeLLM([GOOD] * 2), **kwargs)

    monkeypatch.setattr(ans.RAGPipeline, 'load', staticmethod(fake_load))
    monkeypatch.setattr(ans, 'load_questions', lambda split: questions(2))
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['answers', '--split', 'dev', '--prompt', 'v2', '--sleep', '0'])
    ans.main()
    assert seen['refuse_below'] == float('-inf') and seen['prompt_version'] == 'v2'
    assert (tmp_path / 'answers_dev_fake-fake-1-v2.jsonl').exists()
    assert '2 new' in capsys.readouterr().out


def test_hitting_the_daily_token_allowance_stops_at_once(tmp_path):
    class Daily:
        name, model = 'fake', 'fake-1'

        def __init__(self):
            self.n = 0

        def complete(self, messages):
            self.n += 1
            raise LLMError('Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 198207')
    llm = Daily()
    p = RAGPipeline(StubRetriever(make_results()), llm, refuse_below=float('-inf'), max_retries=0)
    lines = []
    new, _, failed = ans.run(p, questions(6), tmp_path / 'a.jsonl', log=lines.append)
    assert (new, failed, llm.n) == (0, 1, 1)                  # one try, not three, not six
    assert any('daily token allowance' in line for line in lines)


def test_the_source_option_limits_which_questions_are_asked(tmp_path, monkeypatch):
    def fake_load(backend, strategy, **kwargs):
        return RAGPipeline(StubRetriever(make_results()), FakeLLM([GOOD] * 5), **kwargs)

    monkeypatch.setattr(ans.RAGPipeline, 'load', staticmethod(fake_load))
    monkeypatch.setattr(ans, 'load_questions', lambda split: questions(4))      # q0,q2 synthetic; q1,q3 unanswerable
    monkeypatch.setattr(ans, 'ANSWERS', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['answers', '--split', 'dev', '--prompt', 'v1', '--source', 'unanswerable', '--sleep', '0'])
    ans.main()
    rows = saved(next(tmp_path.glob('answers_dev_*.jsonl')))
    assert [r['id'] for r in rows] == ['q1', 'q3']
