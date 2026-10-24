"""Keyword search with BM25: a passage scores higher when it contains the question's words,
especially rare words (a gene name like IDH1 counts for more than the word "study")."""
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# Lowercase letters and digits only. Hyphens split words, so "IDH-wild-type" matches the
# question "IDH wild type", while gene names like IDH1 and p53 stay whole.
TOKEN = re.compile(r'[a-z0-9]+')


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.size = len(texts)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Return up to k (passage_number, score) pairs, best first. Passages that share no word
        with the question are left out instead of being ranked arbitrarily."""
        scores = self.bm25.get_scores(tokenize(query))
        k = min(k, self.size)
        if k <= 0:
            return []
        top = np.argpartition(scores, -k)[-k:]            # fast: no need to sort everything
        top = top[np.argsort(scores[top])[::-1]]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> 'BM25Index':
        with open(path, 'rb') as f:
            return pickle.load(f)
