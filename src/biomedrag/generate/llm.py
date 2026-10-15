"""Talks to the language model that writes the answer.

Two backends, same interface (`complete(messages) -> text`):
  ollama  a model running on your own computer. Free, private, slower. Good for development.
  groq    a hosted model. Fast and stronger. Good for the public demo.

`messages` is a list like [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}].
Problems (server not running, bad key, rate limit) become an `LLMError` with a plain message.
"""
import os
import time
from typing import ClassVar

import httpx

from biomedrag import config


class LLMError(RuntimeError):
    """The model could not be reached or refused the request. The message says what to do."""


def _detail(response: httpx.Response) -> str:
    """The server's own explanation of an error, shortened."""
    try:
        data = response.json()
        err = data.get('error') if isinstance(data, dict) else None
        if isinstance(err, dict):
            err = err.get('message')
        if err:
            return str(err)[:300]
    except ValueError:
        pass
    return response.text[:300]


class OllamaLLM:
    name = 'ollama'

    def __init__(self, model: str | None = None, url: str | None = None,
                 client: httpx.Client | None = None):
        """`client` is only passed in by tests, which use a fake server."""
        self.model = model or config.OLLAMA_MODEL
        self.url = (url or config.OLLAMA_URL).rstrip('/')
        # The first call can take minutes while Ollama loads the model into memory.
        self.client = client or httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    def complete(self, messages: list[dict]) -> str:
        body = {
            'model': self.model, 'messages': messages, 'stream': False,
            # Five abstracts need more room than Ollama's small default context window.
            'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': 700},
        }
        try:
            response = self.client.post(f'{self.url}/api/chat', json=body)
        except httpx.ConnectError as e:
            raise LLMError(f'Cannot reach Ollama at {self.url}. Is it running? Open the Ollama '
                           f'app (or run "ollama serve") and try again.') from e
        except httpx.TimeoutException as e:
            raise LLMError('Ollama took longer than 5 minutes to answer. Close other heavy '
                           'programs, or use a smaller model, or use the groq backend.') from e
        if response.status_code != 200:
            raise LLMError(f'Ollama returned HTTP {response.status_code}: {_detail(response)}. '
                           f'If the model is missing, run: ollama pull {self.model}')
        try:
            return response.json()['message']['content']
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError('Ollama sent a reply in an unexpected format.') from e


class GroqLLM:
    name = 'groq'
    RETRY_ON: ClassVar[set[int]] = {429, 500, 502, 503, 504}        # busy or rate-limited: waiting a moment can fix it

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 url: str | None = None, client: httpx.Client | None = None,
                 sleep=time.sleep, attempts: int = 3):
        """`client` and `sleep` are only passed in by tests."""
        self.model = model or config.GROQ_MODEL
        self.api_key = (api_key if api_key is not None else os.environ.get('GROQ_API_KEY', '')).strip()
        self.url = url or config.GROQ_URL
        self.client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._sleep = sleep
        self.attempts = attempts

    def _wait(self, response: httpx.Response | None, attempt: int) -> float:
        """How long to pause before trying again: what the server asks for, else 2, 4, 8 ... seconds."""
        asked = response.headers.get('retry-after') if response is not None else None
        try:
            return min(max(float(asked), 1.0), 60.0)
        except (TypeError, ValueError):
            return float(2 ** attempt)

    @staticmethod
    def _read_answer(response: httpx.Response) -> str:
        try:
            choice = response.json()['choices'][0]
            text = choice['message'].get('content') or ''
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError('Groq sent a reply in an unexpected format.') from e
        if not text.strip():
            raise LLMError(f'Groq returned an empty answer (finish reason: {choice.get("finish_reason")}). '
                           f'The model probably ran out of tokens while thinking. Try another GROQ_MODEL.')
        return text

    def _explain(self, response: httpx.Response) -> str:
        code = response.status_code
        if code == 404:
            return (f'Groq says the model {self.model!r} is not available to your account (HTTP 404). '
                    f'Set GROQ_MODEL in .env to a model your key can use. {_detail(response)}')
        if code == 401:
            return ('Groq rejected the API key (HTTP 401). Check GROQ_API_KEY in your .env file: '
                    'copy the whole key, with no quotes and no spaces.')
        if code == 429:
            return (f'Groq says too many requests (HTTP 429), still after {self.attempts} tries. '
                    f'Wait a minute and try again. {_detail(response)}')
        return f'Groq returned HTTP {code}: {_detail(response)}'

    def complete(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise LLMError('GROQ_API_KEY is empty. Put your key in the .env file '
                           '(GROQ_API_KEY=...) or use --backend ollama.')
        body = {'model': self.model, 'messages': messages, 'temperature': 0,
                'max_completion_tokens': 700}
        if self.model.startswith('openai/gpt-oss'):
            # These models "think" first and the thinking uses up the token budget. Keep it short,
            # leave room for the answer, and do not send the thinking back to us.
            body.update(reasoning_effort='low', include_reasoning=False, max_completion_tokens=2500)
        headers = {'Authorization': f'Bearer {self.api_key}'}
        for attempt in range(1, self.attempts + 1):
            last = attempt == self.attempts
            try:
                response = self.client.post(self.url, json=body, headers=headers)
            except httpx.TransportError as e:                   # no internet, timeout, ...
                if last:
                    raise LLMError(f'Cannot reach Groq ({type(e).__name__}). '
                                   f'Check your internet connection.') from e
                self._sleep(self._wait(None, attempt))
                continue
            if response.status_code == 200:
                return self._read_answer(response)
            if response.status_code in self.RETRY_ON and not last:
                self._sleep(self._wait(response, attempt))
                continue
            raise LLMError(self._explain(response))
        raise AssertionError('unreachable')      # the loop always returns or raises


def get_llm(backend: str | None = None, **kwargs):
    """Build the model client named by `backend` (default: LLM_BACKEND from .env)."""
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == 'ollama':
        return OllamaLLM(**kwargs)
    if backend == 'groq':
        return GroqLLM(**kwargs)
    raise ValueError(f"unknown backend {backend!r}; choose 'ollama' or 'groq'")
