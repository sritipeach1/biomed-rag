"""Build the search indexes for one chunking strategy.

Run:  python -m biomedrag.index.build --strategy abstract
      python -m biomedrag.index.build --strategy abstract --limit 2000     (quick speed test)
      python -m biomedrag.index.build --strategy abstract --skip-dense     (keyword index only)

Reads   data/chunks_<strategy>.jsonl
Writes  index/<strategy>/chunks.jsonl, bm25.pkl, dense.faiss, meta.json

The embedding step is slow, so progress is saved every few thousand passages. If the run is
interrupted, run the same command again and it continues where it stopped.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from biomedrag import config
from biomedrag.index.bm25 import BM25Index
from biomedrag.index.dense import DenseIndex


def read_chunks(path: Path, limit: int | None = None) -> list[dict]:
    chunks = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break
    return chunks


def fingerprint(chunks: list[dict]) -> str:
    """Identifies exactly which passages are being embedded, to catch stale saved blocks."""
    return hashlib.md5('\n'.join(c['chunk_id'] for c in chunks).encode('utf-8')).hexdigest()


def build_index(strategy: str, limit: int | None = None, skip_dense: bool = False,
                batch_size: int = 64, model=None, chunks_file: Path | None = None,
                index_root: Path = config.INDEX) -> Path:
    """Build and save the indexes. `model` and `chunks_file` are only passed in by tests."""
    name = strategy if not limit else f'{strategy}_n{limit}'
    out = index_root / name
    out.mkdir(parents=True, exist_ok=True)

    chunks = read_chunks(chunks_file or config.chunks_path(strategy), limit)
    if not chunks:
        raise SystemExit(f'No passages found for strategy {strategy!r}. Run the chunking step first.')
    texts = [c['text'] for c in chunks]
    print(f'{len(chunks)} passages ("{strategy}") -> {out}')

    with open(out / 'chunks.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(c) + '\n' for c in chunks)

    t = time.perf_counter()
    BM25Index(texts).save(out / 'bm25.pkl')
    print(f'keyword index saved in {time.perf_counter() - t:.0f}s')

    meta = {'strategy': strategy, 'n_chunks': len(chunks), 'embed_model': config.EMBED_MODEL,
            'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    if not skip_dense:
        dense = DenseIndex(config.EMBED_MODEL, model=model)
        print('loading the embedding model (the first time this downloads about 130 MB) ...')
        _ = dense.model

        parts = out / 'parts'
        marker = parts / 'fingerprint.txt'
        stale = parts.exists() and (not marker.exists() or marker.read_text() != fingerprint(chunks))
        if parts.exists() and (stale or limit):        # speed tests always start fresh
            shutil.rmtree(parts)                       # saved blocks may belong to other passages
        parts.mkdir(exist_ok=True)
        marker.write_text(fingerprint(chunks))

        t = time.perf_counter()
        vectors = dense.embed_resumable(texts, parts, batch_size=batch_size)
        dense.build(vectors)
        dense.save(out / 'dense.faiss')
        seconds = max(time.perf_counter() - t, 1e-9)
        shutil.rmtree(parts)
        print(f'embedding index saved ({vectors.shape[0]} vectors, {vectors.shape[1]} dimensions) '
              f'in {seconds / 60:.1f} min = {len(chunks) / seconds:.0f} passages per second')
        meta['embed_seconds'] = round(seconds)
        if limit:
            total = len(read_chunks(chunks_file or config.chunks_path(strategy)))
            print(f'At this speed the full "{strategy}" build ({total} passages) would take '
                  f'about {seconds * total / len(chunks) / 60:.0f} minutes.')

    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'done: {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description='Build the search indexes for one chunking strategy.')
    ap.add_argument('--strategy', default='abstract', choices=['abstract', 'sentences'])
    ap.add_argument('--limit', type=int, default=None,
                    help='only use the first N passages (a quick speed test, saved to its own folder)')
    ap.add_argument('--skip-dense', action='store_true', help='build only the keyword index')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    build_index(args.strategy, args.limit, args.skip_dense, args.batch_size)


if __name__ == '__main__':
    main()
