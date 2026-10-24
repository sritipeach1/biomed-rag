"""The web API.

    GET  /          the demo page (static/index.html)
    GET  /health    is the service ready?
    GET  /examples  the example questions that have saved answers
    POST /ask       {"question": "..."}  ->  a cited answer, or an honest refusal

Run it:  uvicorn biomedrag.api.app:app --port 8000
"""
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from biomedrag import config
from biomedrag.api.bootstrap import load_pipeline
from biomedrag.api.demo_cache import key, load_cache
from biomedrag.api.limits import DailyBudget, RateLimiter
from biomedrag.api.schemas import AskRequest, AskResponse, present
from biomedrag.generate.llm import LLMError

log = logging.getLogger('biomedrag.api')

HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    # One self-contained page: its own script and style, nothing from other websites.
    'Content-Security-Policy': ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                "style-src 'self' 'unsafe-inline'; base-uri 'none'; form-action 'self'"),
}


def client_ip(request: Request) -> str:
    """Who is asking. Behind a hosting proxy the real address is the LAST entry of X-Forwarded-For
    (the proxy adds it; earlier entries can be written by the visitor)."""
    forwarded = request.headers.get('x-forwarded-for', '')
    if forwarded.strip():
        return forwarded.split(',')[-1].strip()
    return request.client.host if request.client else 'unknown'


def create_app(pipeline=None, loader=load_pipeline, cache: list[dict] | None = None,
               limiter: RateLimiter | None = None, budget: DailyBudget | None = None,
               static_dir: Path = config.STATIC, origins: str = config.ALLOWED_ORIGIN) -> FastAPI:
    """`pipeline`, `cache`, `limiter` and `budget` are only passed in by tests."""
    state: dict = {'pipe': pipeline, 'error': None}
    saved = load_cache(static_dir / 'demo_cache.json') if cache is None else cache
    by_question = {key(item['question']): item['response'] for item in saved}
    limiter = limiter or RateLimiter(config.RATE_PER_MINUTE)
    budget = budget or DailyBudget(config.DAILY_ANSWER_LIMIT)
    lock = threading.Lock()                     # one question at a time: the free CPU is small

    def warm_up() -> None:
        try:
            state['pipe'] = loader()
            log.info('pipeline ready')
        except Exception as e:
            state['error'] = f'{type(e).__name__}: {e}'
            log.exception('could not load the pipeline')

    @asynccontextmanager
    async def lifespan(_app):
        if state['pipe'] is None and loader is not None:
            threading.Thread(target=warm_up, daemon=True).start()     # the server answers /health meanwhile
        yield

    app = FastAPI(title='BiomedRAG', description='Cited answers from PubMed glioma abstracts.',
                  version='1.0.0', lifespan=lifespan)
    app.state.api = state
    origin_list = [o.strip() for o in origins.split(',') if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origin_list, allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type'])

    @app.middleware('http')
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get('/health')
    def health():
        status = 'error' if state['error'] else ('ok' if state['pipe'] is not None else 'loading')
        return {'status': status, 'ready': status == 'ok'}

    @app.get('/examples')
    def examples():
        return [{'question': item['question'], 'answered': item['response']['answered']} for item in saved]

    @app.post('/ask', response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        hit = by_question.get(key(body.question))
        if hit is not None:                                     # a saved example: free and instant
            return {**hit, 'from_cache': True}
        if state['error']:
            raise HTTPException(503, 'The search index could not be loaded, so questions cannot be '
                                     'answered right now. The example questions still work.')
        if state['pipe'] is None:
            raise HTTPException(503, 'The service is still starting up. Please try again in a minute.')
        if not limiter.allow(client_ip(request)):
            raise HTTPException(429, 'Too many questions in a minute. Please wait a little and try again.',
                                headers={'Retry-After': '60'})
        if budget.exhausted():
            raise HTTPException(429, "Today's free allowance for this demo is used up. The example "
                                     'questions still work, and it resets tomorrow (UTC).')
        try:
            with lock:
                answer = state['pipe'].answer(body.question)
        except LLMError as e:
            log.warning('language model error: %s', e)
            raise HTTPException(503, 'The language model is busy or unavailable (the free tier has '
                                     'limits). Please try again in a minute, or use an example.') from e
        if answer.llm_seconds > 0:                              # only real model calls use the allowance
            budget.record()
        return present(answer)

    @app.get('/', include_in_schema=False)
    def home():
        page = static_dir / 'index.html'
        if not page.exists():
            raise HTTPException(404, 'The demo page is missing.')
        return FileResponse(page, media_type='text/html')

    return app


app = create_app()
