import json
from datetime import date

import pytest
from fakes import FakeLLM, StubRetriever, make_results

from biomedrag.api import bootstrap, demo_cache
from biomedrag.api.limits import DailyBudget, RateLimiter
from biomedrag.generate.pipeline import RAGPipeline

# ---------- limits ----------


def test_rate_limiter_allows_the_quota_then_blocks_then_recovers():
    now = [0.0]
    lim = RateLimiter(2, clock=lambda: now[0])
    assert lim.allow('a') and lim.allow('a') and not lim.allow('a')
    assert lim.allow('b')                                 # someone else is unaffected
    now[0] = 59.9
    assert not lim.allow('a')
    now[0] = 60.1
    assert lim.allow('a')


def test_rate_limiter_forgets_quiet_visitors():
    now = [0.0]
    lim = RateLimiter(1, clock=lambda: now[0])
    for i in range(10):
        lim.allow(f'v{i}')
    now[0] = 1000.0
    for _ in range(500):
        lim.allow('busy')
    assert set(lim.hits) == {'busy'}


def test_daily_budget_counts_and_resets_at_midnight_utc():
    day = [date(2026, 10, 24)]
    b = DailyBudget(2, today=lambda: day[0])
    assert not b.exhausted()
    b.record()
    b.record()
    assert b.exhausted()
    day[0] = date(2026, 10, 25)
    assert not b.exhausted() and b.used == 0


# ---------- bootstrap ----------


def make_files(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in bootstrap.NEEDED:
        (folder / name).write_text('x')


def test_ensure_index_does_nothing_when_the_files_exist(tmp_path):
    make_files(tmp_path / 'abstract')
    calls = []
    out = bootstrap.ensure_index('abstract', tmp_path, repo='me/idx', download=lambda **kw: calls.append(kw))
    assert out == tmp_path / 'abstract' and calls == []


def test_ensure_index_downloads_when_missing(tmp_path):
    calls = []

    def fake_download(**kw):
        calls.append(kw)
        make_files(tmp_path / 'abstract')

    out = bootstrap.ensure_index('abstract', tmp_path, repo='me/idx', token='tok', download=fake_download)
    assert out == tmp_path / 'abstract'
    assert calls == [{'repo_id': 'me/idx', 'repo_type': 'dataset', 'local_dir': str(tmp_path / 'abstract'),
                      'token': 'tok'}]


def test_ensure_index_explains_what_to_do_without_a_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.config, 'INDEX_REPO', '')
    with pytest.raises(RuntimeError, match='INDEX_REPO'):
        bootstrap.ensure_index('abstract', tmp_path)


def test_ensure_index_notices_an_incomplete_download(tmp_path):
    with pytest.raises(RuntimeError, match='still missing'):
        bootstrap.ensure_index('abstract', tmp_path, repo='me/idx', download=lambda **kw: None)


# ---------- demo cache ----------


def test_read_questions_skips_comments_and_blank_lines(tmp_path):
    f = tmp_path / 'q.txt'
    f.write_text('# note\n\nFirst question?\n  Second question?  \n', encoding='utf-8')
    assert demo_cache.read_questions(f) == ['First question?', 'Second question?']


def test_build_saves_the_public_view_and_load_reads_it_back(tmp_path):
    pipe = RAGPipeline(StubRetriever(make_results()), FakeLLM(['IDH helps [PMID:1].']), refuse_below=3.0)
    path = tmp_path / 'cache.json'
    log = []
    assert demo_cache.build(pipe, ['Does IDH matter in glioma?'], path, log=log.append) == 1
    items = demo_cache.load_cache(path)
    assert items[0]['question'] == 'Does IDH matter in glioma?'
    assert items[0]['response']['answered'] and items[0]['response']['sources'][0]['pmid'] == '1'
    assert 'text' not in items[0]['response']['sources'][0]
    assert json.loads(path.read_text(encoding='utf-8')) == items and log


def test_load_cache_of_a_missing_file_is_empty(tmp_path):
    assert demo_cache.load_cache(tmp_path / 'none.json') == []


def test_key_ignores_case_and_spacing():
    assert demo_cache.key('  Does  IDH\nMATTER? ') == 'does idh matter?'
