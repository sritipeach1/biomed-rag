from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
INDEX = ROOT / 'index'

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


def chunks_path(strategy: str) -> Path:
    return DATA / f'chunks_{strategy}.jsonl'


def index_dir(name: str) -> Path:
    return INDEX / name
