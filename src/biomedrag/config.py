from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
INDEX = ROOT / 'index'

CORPUS_PATH = DATA / 'abstracts.jsonl'
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
