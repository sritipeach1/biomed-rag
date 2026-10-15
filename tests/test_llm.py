import json

import httpx
import pytest

from biomedrag.generate.llm import GroqLLM, LLMError, OllamaLLM, get_llm

MESSAGES = [{'role': 'user', 'content': 'hi'}]


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- Ollama ----------

def test_ollama_sends_the_request_we_expect_and_returns_the_text():
    seen = {}

    def handler(request):
        seen['url'] = str(request.url)
        seen['body'] = json.loads(request.content)
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'hello'}})

    llm = OllamaLLM(model='tiny', url='http://box:1234/', client=client_for(handler))
    assert llm.complete(MESSAGES) == 'hello'
    assert seen['url'] == 'http://box:1234/api/chat'
    body = seen['body']
    assert body['model'] == 'tiny' and body['stream'] is False and body['messages'] == MESSAGES
    assert body['options']['temperature'] == 0
    assert body['options']['num_ctx'] >= 8192


def test_ollama_not_running_gives_a_helpful_message():
    def handler(request):
        raise httpx.ConnectError('refused')

    with pytest.raises(LLMError, match='Is it running'):
        OllamaLLM(client=client_for(handler)).complete(MESSAGES)


def test_ollama_missing_model_says_how_to_pull_it():
    def handler(request):
        return httpx.Response(404, json={'error': "model 'nope' not found"})

    with pytest.raises(LLMError, match='ollama pull nope') as e:
        OllamaLLM(model='nope', client=client_for(handler)).complete(MESSAGES)
    assert 'not found' in str(e.value)


def test_ollama_slow_reply_becomes_a_timeout_message():
    def handler(request):
        raise httpx.ReadTimeout('slow')

    with pytest.raises(LLMError, match='longer than 5 minutes'):
        OllamaLLM(client=client_for(handler)).complete(MESSAGES)


def test_ollama_odd_reply_shape_is_reported():
    with pytest.raises(LLMError, match='unexpected format'):
        OllamaLLM(client=client_for(lambda r: httpx.Response(200, json={'oops': 1}))).complete(MESSAGES)


# ---------- Groq ----------

def ok_reply(text='answer'):
    return httpx.Response(200, json={'choices': [{'message': {'content': text}}]})


def test_groq_sends_the_key_and_model_and_returns_the_text():
    seen = {}

    def handler(request):
        seen['auth'] = request.headers.get('authorization')
        seen['body'] = json.loads(request.content)
        return ok_reply('hi there')

    llm = GroqLLM(model='m1', api_key='gsk_secret', client=client_for(handler))
    assert llm.complete(MESSAGES) == 'hi there'
    assert seen['auth'] == 'Bearer gsk_secret'
    assert seen['body']['model'] == 'm1' and seen['body']['temperature'] == 0
    assert seen['body']['messages'] == MESSAGES


def test_groq_without_a_key_fails_before_any_request():
    def handler(request):
        raise AssertionError('no request should be sent')

    with pytest.raises(LLMError, match='GROQ_API_KEY is empty'):
        GroqLLM(api_key='  ', client=client_for(handler)).complete(MESSAGES)


def test_groq_waits_as_asked_then_retries_after_a_rate_limit():
    replies = [httpx.Response(429, headers={'retry-after': '7'}), ok_reply('finally')]
    slept = []
    llm = GroqLLM(api_key='k', client=client_for(lambda r: replies.pop(0)), sleep=slept.append)
    assert llm.complete(MESSAGES) == 'finally'
    assert slept == [7.0]


def test_groq_gives_up_after_the_allowed_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={'error': {'message': 'slow down'}})

    llm = GroqLLM(api_key='k', client=client_for(handler), sleep=lambda s: None, attempts=3)
    with pytest.raises(LLMError, match='too many requests'):
        llm.complete(MESSAGES)
    assert len(calls) == 3


def test_groq_bad_key_is_not_retried_and_the_key_is_never_printed():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={'error': {'message': 'Invalid API Key'}})

    llm = GroqLLM(api_key='gsk_topsecret', client=client_for(handler), sleep=lambda s: None)
    with pytest.raises(LLMError, match='rejected the API key') as e:
        llm.complete(MESSAGES)
    assert len(calls) == 1
    assert 'gsk_topsecret' not in str(e.value)


def test_groq_retries_a_dropped_connection_then_succeeds():
    outcomes = [httpx.ConnectError('down'), ok_reply('back')]

    def handler(request):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    llm = GroqLLM(api_key='k', client=client_for(handler), sleep=lambda s: None)
    assert llm.complete(MESSAGES) == 'back'


def test_groq_unknown_model_error_names_the_model_and_the_fix():
    def handler(request):
        return httpx.Response(404, json={'error': {'message': 'model does not exist'}})

    with pytest.raises(LLMError, match='model does not exist') as e:
        GroqLLM(model='old-model', api_key='k', client=client_for(handler)).complete(MESSAGES)
    assert "'old-model'" in str(e.value) and 'GROQ_MODEL' in str(e.value)


def sent_body(model):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return ok_reply()

    GroqLLM(model=model, api_key='k', client=client_for(handler)).complete(MESSAGES)
    return seen


def test_thinking_models_get_a_short_thinking_budget_and_more_room():
    body = sent_body('openai/gpt-oss-120b')
    assert body['reasoning_effort'] == 'low'
    assert body['include_reasoning'] is False
    assert body['max_completion_tokens'] >= 2000


def test_other_models_do_not_get_thinking_settings():
    body = sent_body('some/plain-model')
    assert 'reasoning_effort' not in body and 'include_reasoning' not in body
    assert body['max_completion_tokens'] == 700


@pytest.mark.parametrize('message', [{'content': ''}, {'content': None}, {'content': '   '}, {}])
def test_groq_empty_answers_are_reported_not_passed_on(message):
    def handler(request):
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': 'length'}]})

    with pytest.raises(LLMError, match='empty answer.*length'):
        GroqLLM(api_key='k', client=client_for(handler)).complete(MESSAGES)


# ---------- choosing a backend ----------

def test_get_llm_picks_the_backend_by_name(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'abc')
    assert isinstance(get_llm('ollama'), OllamaLLM)
    groq = get_llm('GROQ')
    assert isinstance(groq, GroqLLM) and groq.api_key == 'abc'


def test_get_llm_rejects_unknown_backends():
    with pytest.raises(ValueError, match='unknown backend'):
        get_llm('skynet')
