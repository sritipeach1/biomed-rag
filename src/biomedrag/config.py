import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')      # secrets and settings live in .env, which git never uploads


def _env(name: str, default: str = '') -> str:
    """Read a setting from .env or the environment. A blank value counts as "not set"."""
    return os.environ.get(name, '').strip() or default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f'{name} in .env must be a number, but it is {value!r}') from None


DATA = ROOT / 'data'
INDEX = ROOT / 'index'
EVAL = ROOT / 'eval'                         # question sets and result tables (committed to git)
RESULTS = EVAL / 'results'
QUESTIONS_PATH = EVAL / 'questions.jsonl'

CORPUS_PATH = DATA / 'abstracts.jsonl'      # everything fetched from PubMed (Step 1)
SAMPLE_PATH = DATA / 'docs_sample.jsonl'    # cleaned random sample: the corpus we index (Step 2)

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# bge models work best when the *question* (not the passages) starts with this instruction.
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Cleaning and sampling (Step 2)
MIN_WORDS = 50          # shorter abstracts carry too little information
MAX_WORDS = 600         # longer ones are usually not real abstracts and get cut off by the embedder
SAMPLE_SIZE = 30_000    # how many abstracts we index; keeps embedding time reasonable
SEED = 7                # fixed seed so everyone gets the same sample

# Retrieval (Step 3)
K_CANDIDATES = 50       # how many results each search engine returns before fusion
K_RERANK = 20           # how many fused candidates the reranker re-scores
K_FINAL = 5             # how many papers go to the answer step
RRF_K = 60              # standard constant for reciprocal rank fusion
MODES = ('bm25', 'dense', 'hybrid', 'hybrid_rerank')
EMBED_BLOCK = 10_000    # texts embedded per saved block, so an interrupted build can resume

# Answering (Step 4)
LLM_BACKEND = _env('LLM_BACKEND', 'ollama').lower()     # 'ollama' (on your PC) or 'groq' (hosted)
OLLAMA_URL = _env('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = _env('OLLAMA_MODEL', 'qwen2.5:3b')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = _env('GROQ_MODEL', 'openai/gpt-oss-120b')     # the models a key may use differ per account
# If the best reranker score is below this, the question is treated as outside the literature and
# no answer is written. 5.0 was chosen on the dev questions in Step 5 (the plateau was about 4.4 to 5.4).
# The score scale belongs to this reranker model: change the model, redo the threshold.
REFUSE_BELOW = _env_float('REFUSE_BELOW', 5.0)
# Which instructions the model gets (see generate/prompt.py). v3 was chosen on the dev questions in Step 5.
PROMPT_VERSION = _env('PROMPT_VERSION', 'v3')
MAX_RETRIES = 1         # extra attempts when the model's citations are missing or invalid


def chunks_path(strategy: str) -> Path:
    return DATA / f'chunks_{strategy}.jsonl'


def index_dir(name: str) -> Path:
    return INDEX / name
