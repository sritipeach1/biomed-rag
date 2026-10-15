"""The numbers: ranking metrics, bootstrap confidence intervals, and the refusal-threshold sweep.
Plain functions with no models and no files, so they are easy to test by hand."""
import math

import numpy as np


def first_gold_rank(pmids: list[str], gold: list[str]) -> int | None:
    """1-based position of the first gold paper in a ranked list of PMIDs, or None if absent."""
    wanted = set(gold)
    for rank, pmid in enumerate(pmids, start=1):
        if pmid in wanted:
            return rank
    return None


def hits(ranks: list[int | None], k: int) -> list[float]:
    """1.0 for every question whose gold paper is within the top k, else 0.0."""
    return [1.0 if r is not None and r <= k else 0.0 for r in ranks]


def reciprocal_ranks(ranks: list[int | None], cutoff: int = 10) -> list[float]:
    """1/rank, or 0 if the gold paper is missing or below the cutoff."""
    return [1.0 / r if r is not None and r <= cutoff else 0.0 for r in ranks]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float('nan')


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    """95% confidence interval of the mean, by resampling the questions with replacement."""
    if not values:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = data[rng.integers(0, len(data), size=(n_boot, len(data)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 7,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) over the same questions, with a 95% interval. If the interval does not
    contain 0, the difference is unlikely to be luck of which questions were drawn."""
    if len(a) != len(b):
        raise ValueError('both systems must be measured on the same questions')
    diff = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diff, n_boot, seed, alpha)
    return mean(diff), lo, hi


def sweep_thresholds(answerable_scores: list[float | None],
                     unanswerable_scores: list[float | None]) -> list[dict]:
    """For every possible cutoff t ("refuse when the best paper scores below t"), the share of
    answerable questions still answered and of unanswerable questions refused.
    A missing score (nothing retrieved) always counts as refused."""
    a = [-math.inf if s is None else s for s in answerable_scores]
    u = [-math.inf if s is None else s for s in unanswerable_scores]
    finite = sorted({s for s in a + u if math.isfinite(s)})
    if not finite:
        return []
    cutoffs = [finite[0] - 1.0, *finite, finite[-1] + 1.0]       # answer everything ... refuse everything
    return [{'threshold': t,
             'answered': mean([1.0 if s >= t else 0.0 for s in a]),
             'refused': mean([1.0 if s < t else 0.0 for s in u])} for t in cutoffs]


def choose_threshold(rows: list[dict], min_refuse: float = 0.8,
                     min_answer: float = 0.85) -> tuple[dict | None, bool]:
    """Pick a row of the sweep. Among cutoffs that meet both targets take the one with the highest
    answered + refused. If none meets both, return the best compromise and met=False: that is a
    real finding to report, not something to hide."""
    if not rows:
        return None, False
    ok = [r for r in rows if r['answered'] >= min_answer and r['refused'] >= min_refuse]
    pool, met = (ok, True) if ok else (rows, False)
    best = max(pool, key=lambda r: (round(r['answered'] + r['refused'], 12), -r['threshold']))
    return best, met
