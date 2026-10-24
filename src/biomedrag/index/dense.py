"""Meaning-based search: each passage becomes a vector (an "embedding"), and a question finds the
passages whose vectors point in the most similar direction, even when no words are shared."""
from pathlib import Path

import faiss
import numpy as np

from biomedrag import config


class DenseIndex:
    def __init__(self, model_name: str = config.EMBED_MODEL, model=None):
        """`model` is only passed in by tests, which use a small fake instead of the real one."""
        self.model_name = model_name
        self._model = model
        self.index = None

    @property
    def model(self):
        if self._model is None:                       # load the real model only when needed
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress)
        return np.asarray(vecs, dtype='float32')

    def embed_resumable(self, texts: list[str], parts_dir: Path, block: int = config.EMBED_BLOCK,
                        batch_size: int = 64) -> np.ndarray:
        """Embed in blocks and save each one, so a crash or a closed laptop loses at most one block.
        Running it again skips the blocks that are already saved."""
        parts_dir.mkdir(parents=True, exist_ok=True)
        n_blocks = (len(texts) + block - 1) // block
        for b in range(n_blocks):
            part = parts_dir / f'emb_{b:05d}.npy'
            if part.exists():
                print(f'  block {b + 1}/{n_blocks}: already done, skipping')
                continue
            print(f'  block {b + 1}/{n_blocks}: embedding {len(texts[b * block:(b + 1) * block])} passages')
            vecs = self.embed(texts[b * block:(b + 1) * block], batch_size, show_progress=True)
            tmp = part.with_suffix('.tmp.npy')        # write to a temp name first, so a half-written
            np.save(tmp, vecs)                        # file is never mistaken for a finished block
            tmp.replace(part)
        return np.concatenate([np.load(parts_dir / f'emb_{b:05d}.npy') for b in range(n_blocks)])

    def build(self, vectors: np.ndarray) -> None:
        # Vectors are normalised to length 1, so inner product equals cosine similarity.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype='float32'))

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        q = self.embed([config.QUERY_PREFIX + query])
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> 'DenseIndex':
        self.index = faiss.read_index(str(path))
        return self
