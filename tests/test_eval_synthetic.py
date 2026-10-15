import json
import sys

import pytest
from fakes import FakeLLM

from biomedrag.eval import make_synthetic as ms
from biomedrag.generate.llm import LLMError

DOCS = [{'pmid': str(100 + i), 'title': f'Paper number {i} on tumours', 'abstract': f'Abstract text {i}.'}
        for i in range(12)]
GOOD = 'How does a new treatment change survival in adult patients?'


@pytest.mark.parametrize('raw, expected', [
    ('How does X work?', 'How does X work?'),
    ('  "How does X work?"  ', 'How does X work?'),
    ('Question: How does X work?', 'How does X work?'),
    ('How does X work?\nSome extra explanation.', 'How does X work?'),
    ('', ''),
])
def test_clean_question(raw, expected):
    assert ms.clean_question(raw) == expected


@pytest.mark.parametrize('question, reason', [
    (GOOD, None),
    ('Why?', 'length'),
    ('What ' + 'very ' * 40 + 'long question?', 'length'),
    ('How does the treatment change survival in adults', 'question mark'),
    ('What did this study find about survival in adults?', 'points at the abstract'),
    ('What do the authors conclude about survival in adults?', 'points at the abstract'),
    ('Do these patients survive longer after treatment?', 'points at the abstract'),
])
def test_rejection_reasons(question, reason):
    got = ms.rejection_reason(question, 'An unrelated title')
    assert (got is None) if reason is None else (reason in got)


def test_a_question_that_copies_the_title_is_rejected():
    title = 'Effect of temozolomide dose on survival in adult glioblastoma patients'
    assert 'title' in ms.rejection_reason('Effect of temozolomide dose on survival in adult glioblastoma patients?', title)


def test_generation_stops_at_n_and_keeps_the_gold_paper(tmp_path):
    llm = FakeLLM([GOOD] * 10)
    rows = ms.generate(llm, DOCS, 3, cache_path=tmp_path / 'cache.jsonl')
    assert len(rows) == 3 and len(llm.calls) == 3
    for r in rows:
        assert r['source'] == 'synthetic' and r['answerable'] is True
        assert r['id'] == 's' + r['gold_pmids'][0] and len(r['gold_pmids']) == 1


def test_bad_questions_are_skipped_and_remembered(tmp_path):
    cache = tmp_path / 'cache.jsonl'
    llm = FakeLLM(['no question mark here at all really', GOOD, GOOD])
    rows = ms.generate(llm, DOCS, 2, cache_path=cache)
    assert len(rows) == 2 and len(llm.calls) == 3
    cached = [json.loads(line) for line in cache.read_text().splitlines()]
    assert [c['accepted'] for c in cached] == [False, True, True]
    assert cached[0]['reason']


def test_running_again_reuses_the_cache_and_asks_the_model_nothing(tmp_path):
    cache = tmp_path / 'cache.jsonl'
    first = ms.generate(FakeLLM([GOOD] * 5), DOCS, 4, cache_path=cache)
    again_llm = FakeLLM([])
    again = ms.generate(again_llm, DOCS, 4, cache_path=cache)
    assert again_llm.calls == [] and again == first


def test_a_larger_n_only_pays_for_the_new_questions(tmp_path):
    cache = tmp_path / 'cache.jsonl'
    ms.generate(FakeLLM([GOOD] * 5), DOCS, 3, cache_path=cache)
    llm = FakeLLM([GOOD] * 5)
    rows = ms.generate(llm, DOCS, 5, cache_path=cache)
    assert len(rows) == 5 and len(llm.calls) == 2


def test_progress_survives_a_model_failure(tmp_path):
    class Flaky(FakeLLM):
        def complete(self, messages):
            if len(self.calls) == 2:
                raise LLMError('rate limit')
            return super().complete(messages)

    cache = tmp_path / 'cache.jsonl'
    with pytest.raises(LLMError):
        ms.generate(Flaky([GOOD] * 5), DOCS, 5, cache_path=cache)
    assert len(cache.read_text().splitlines()) == 2             # the two finished ones were saved
    llm = FakeLLM([GOOD] * 5)
    assert len(ms.generate(llm, DOCS, 5, cache_path=cache)) == 5 and len(llm.calls) == 3


def test_it_gives_up_after_enough_attempts_if_nothing_passes(tmp_path):
    llm = FakeLLM(['bad'] * 100)
    assert ms.generate(llm, DOCS, 2, cache_path=tmp_path / 'c.jsonl', max_attempts_factor=3) == []
    assert len(llm.calls) == 6


def test_the_prompt_contains_the_title_and_abstract(tmp_path):
    llm = FakeLLM([GOOD])
    ms.generate(llm, DOCS, 1, cache_path=tmp_path / 'c.jsonl')
    user = llm.calls[0][1]['content']
    assert 'Paper number' in user and 'Abstract text' in user


def test_the_command_will_not_overwrite_your_edited_file(tmp_path, monkeypatch):
    existing = tmp_path / 'synthetic.jsonl'
    existing.write_text('my careful edits\n')
    monkeypatch.setattr(ms, 'OUT', existing)
    monkeypatch.setattr(sys, 'argv', ['make_synthetic', '--n', '3'])
    with pytest.raises(SystemExit, match='already exists'):
        ms.main()
    assert existing.read_text() == 'my careful edits\n'


def test_the_command_writes_the_file(tmp_path, monkeypatch):
    out = tmp_path / 'synthetic.jsonl'
    monkeypatch.setattr(ms, 'OUT', out)
    monkeypatch.setattr(ms, 'CACHE', tmp_path / 'cache.jsonl')
    monkeypatch.setattr(ms, 'get_llm', lambda backend=None: FakeLLM([GOOD] * 10))
    monkeypatch.setattr(ms, 'load_sample_docs', lambda: iter(DOCS))
    monkeypatch.setattr(sys, 'argv', ['make_synthetic', '--n', '3'])
    ms.main()
    assert len(out.read_text().splitlines()) == 3
