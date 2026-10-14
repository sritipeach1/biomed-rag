from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
INDEX = ROOT / 'index'

CORPUS_PATH = DATA / 'abstracts.jsonl'      # everything fetched from PubMed (Step 1)
SAMPLE_PATH = DATA / 'docs_sample.jsonl'    # cleaned random sample: the corpus we index (Step 2)

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Cleaning and sampling (Step 2)
MIN_WORDS = 50          # shorter abstracts carry too little information
MAX_WORDS = 600         # longer ones are usually not real abstracts and get cut off by the embedder
SAMPLE_SIZE = 30_000    # how many abstracts we index; keeps embedding time reasonable
SEED = 7                # fixed seed so everyone gets the same sample


def chunks_path(strategy: str) -> Path:
    return DATA / f'chunks_{strategy}.jsonl'
