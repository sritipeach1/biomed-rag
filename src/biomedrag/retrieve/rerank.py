"""A cross-encoder reads the question and one passage *together* and scores how well the passage
answers it. That is slower but much more accurate than comparing separate vectors, so we only use
it on the few candidates that the two fast searches already picked."""
from biomedrag import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL, model=None):
        """`model` is only passed in by tests, which use a small fake instead of the real one."""
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top: int = config.K_FINAL) -> list[tuple[dict, float]]:
        """Return the `top` best (chunk, score) pairs. The scale of the scores depends on the model
        and library version (they may be unbounded or squeezed into 0 to 1), so they are not
        probabilities: only their order, and a threshold you calibrate yourself, mean anything."""
        if not chunks:
            return []
        scores = self.model.predict([(query, c['text']) for c in chunks], batch_size=32)
        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:top]
        return [(chunks[i], float(scores[i])) for i in order]
