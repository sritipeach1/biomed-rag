import json

import pytest
from fakes import FakeLLM, StubRetriever, make_results
from fastapi.testclient import TestClient

from biomedrag.api import app as api_app
from biomedrag.api.limits import DailyBudget, RateLimiter
from biomedrag.api.schemas import normalise_citations
from biomedrag.generate.llm import LLMError
from biomedrag.generate.pipeline import RAGPipeline

GOOD = 'IDH mutation predicts better survival 【PMID:1】. MGMT predicts response [PMID:2].'
SAVED = [{'question': 'Does IDH matter in glioma?',
          'response': {'question': 'Does IDH matter in glioma?', 'status': 'answered', 'answered': True,
                       'text': 'Yes [PMID:1].', 'sources': [], 'seconds': 9.0, 'from_cache': False}}]


class BrokenLLM:
    name, model = 'broken', 'broken-1'

    def complete(self, messages):
        raise LLMError('rate limited')


def make_pipe(replies=(GOOD,), scores=(7.5, 6.0, 4.5), llm=None):
    return RAGPipeline(StubRetriever(make_results(scores)), llm or FakeLLM(list(replies)), refuse_below=3.0)


def client(pipe=None, cache=None, limiter=None, budget=None, **kwargs):
    app = api_app.create_app(pipeline=pipe, loader=None, cache=SAVED if cache is None else cache,
                             limiter=limiter, budget=budget, **kwargs)
    return TestClient(app)


QUESTION = {'question': 'What does IDH mutation mean for survival?'}


def test_health_says_loading_ok_or_error():
    assert client().get('/health').json() == {'status': 'loading', 'ready': False}
    assert client(make_pipe()).get('/health').json() == {'status': 'ok', 'ready': True}
    c = client()
    c.app.state.api['error'] = 'boom'
    assert c.get('/health').json() == {'status': 'error', 'ready': False}


def test_a_normal_question_returns_a_cited_answer_with_clean_brackets_and_links():
    r = client(make_pipe()).post('/ask', json=QUESTION)
    assert r.status_code == 200
    body = r.json()
    assert body['answered'] and body['status'] == 'answered' and body['from_cache'] is False
    assert '【' not in body['text'] and '[PMID:1]' in body['text']
    assert [(s['pmid'], s['cited']) for s in body['sources']] == [('1', True), ('2', True), ('3', False)]
    assert body['sources'][0]['url'] == 'https://pubmed.ncbi.nlm.nih.gov/1/'


def test_the_abstract_text_is_never_sent():
    body = client(make_pipe()).post('/ask', json=QUESTION).json()
    assert 'text' not in body['sources'][0]
    assert 'predicts better survival in glioma patients' not in json.dumps(body['sources'])


def test_a_refusal_is_returned_as_a_normal_response_with_no_sources():
    r = client(make_pipe(replies=[], scores=(1.0, 0.5, 0.2))).post('/ask', json=QUESTION)
    body = r.json()
    assert r.status_code == 200 and not body['answered'] and body['status'] == 'refused_low_score'
    assert body['sources'] == []


@pytest.mark.parametrize('question', ['hi', '       ', 'x' * 501, ''])
def test_bad_questions_are_rejected_with_422(question):
    assert client(make_pipe()).post('/ask', json={'question': question}).status_code == 422


def test_missing_question_is_422():
    assert client(make_pipe()).post('/ask', json={}).status_code == 422


def test_a_saved_example_is_answered_from_the_file_without_the_model():
    llm = FakeLLM([])
    c = client(make_pipe(llm=llm))
    r = c.post('/ask', json={'question': '  does IDH matter   in glioma? '})
    assert r.status_code == 200 and r.json()['from_cache'] is True and r.json()['text'] == 'Yes [PMID:1].'
    assert llm.calls == []


def test_saved_examples_still_work_while_the_service_is_starting_or_broken():
    c = client()                                          # no pipeline yet
    assert c.post('/ask', json={'question': 'Does IDH matter in glioma?'}).status_code == 200
    r = c.post('/ask', json=QUESTION)
    assert r.status_code == 503 and 'starting' in r.json()['detail']
    c.app.state.api['error'] = 'index missing'
    r = c.post('/ask', json=QUESTION)
    assert r.status_code == 503 and 'index' in r.json()['detail'] and 'missing' not in r.json()['detail']


def test_the_rate_limit_blocks_a_busy_visitor_but_not_saved_examples():
    c = client(make_pipe(replies=[GOOD, GOOD]), limiter=RateLimiter(2))
    assert c.post('/ask', json=QUESTION).status_code == 200
    assert c.post('/ask', json=QUESTION).status_code == 200
    r = c.post('/ask', json=QUESTION)
    assert r.status_code == 429 and r.headers['retry-after'] == '60'
    assert c.post('/ask', json={'question': 'Does IDH matter in glioma?'}).status_code == 200


def test_visitors_are_told_apart_by_the_last_forwarded_address():
    c = client(make_pipe(replies=[GOOD, GOOD, GOOD]), limiter=RateLimiter(1))
    assert c.post('/ask', json=QUESTION, headers={'x-forwarded-for': 'spoofed, 10.0.0.1'}).status_code == 200
    assert c.post('/ask', json=QUESTION, headers={'x-forwarded-for': 'other-spoof, 10.0.0.1'}).status_code == 429
    assert c.post('/ask', json=QUESTION, headers={'x-forwarded-for': 'x, 10.0.0.2'}).status_code == 200


def test_the_daily_allowance_counts_model_calls_only():
    budget = DailyBudget(1)
    c = client(make_pipe(replies=[GOOD, GOOD]), budget=budget, limiter=RateLimiter(50))
    assert c.post('/ask', json=QUESTION).status_code == 200
    r = c.post('/ask', json=QUESTION)
    assert r.status_code == 429 and 'allowance' in r.json()['detail']
    refusing = client(make_pipe(replies=[], scores=(0.1, 0.1, 0.1)), budget=DailyBudget(1))
    for _ in range(3):                                    # refusals before the model are free
        assert refusing.post('/ask', json=QUESTION).json()['status'] == 'refused_low_score'


def test_a_model_failure_becomes_a_friendly_503_without_the_internal_message():
    r = client(make_pipe(llm=BrokenLLM())).post('/ask', json=QUESTION)
    assert r.status_code == 503 and 'rate limited' not in r.json()['detail']


def test_examples_lists_the_saved_questions():
    assert client().get('/examples').json() == [{'question': 'Does IDH matter in glioma?', 'answered': True}]


def test_the_home_page_is_served_with_safety_headers(tmp_path):
    (tmp_path / 'index.html').write_text('<h1>hi</h1>', encoding='utf-8')
    r = client(static_dir=tmp_path).get('/')
    assert r.status_code == 200 and '<h1>hi</h1>' in r.text
    assert r.headers['x-content-type-options'] == 'nosniff' and "default-src 'self'" in r.headers['content-security-policy']
    assert client(static_dir=tmp_path / 'nope').get('/').status_code == 404


def test_the_shipped_page_has_the_disclaimer_and_builds_text_safely():
    page = (api_app.config.STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'not medical advice' in page
    assert 'innerHTML' not in page                        # answers are inserted as text, never as HTML


def test_the_page_loads_the_background_pipeline_on_startup():
    started = []

    def loader():
        started.append(True)
        return make_pipe()

    app = api_app.create_app(pipeline=None, loader=loader, cache=[])
    with TestClient(app) as c:                            # entering the block runs the startup code
        for _ in range(100):
            if c.get('/health').json()['ready']:
                break
            import time
            time.sleep(0.02)
        assert c.get('/health').json()['ready'] and started == [True]


def test_a_failing_loader_is_reported_in_health_not_raised():
    def loader():
        raise RuntimeError('no index')

    app = api_app.create_app(pipeline=None, loader=loader, cache=[])
    with TestClient(app) as c:
        for _ in range(100):
            if c.get('/health').json()['status'] == 'error':
                break
            import time
            time.sleep(0.02)
        assert c.get('/health').json()['status'] == 'error'


def test_normalise_citations_only_changes_the_brackets():
    assert normalise_citations('A 【PMID:1】 and [PMID:2].') == 'A [PMID:1] and [PMID:2].'
