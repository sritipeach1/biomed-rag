from biomedrag import config


def rrf(rankings: list[list[int]], k: int = config.RRF_K, top: int = config.K_CANDIDATES) -> list[int]:
    """Reciprocal Rank Fusion: merge several ranked lists into one.

    A passage earns 1 / (k + rank) points from every list it appears in, so one that ranks well in
    both the keyword list and the meaning list beats one that is top in only a single list. Only
    ranks are used, so the lists' scores never have to be comparable."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top]


def collapse_to_pmid(chunks: list[dict]) -> list[dict]:
    """Keep only the best-ranked passage of each paper (the input must already be best first).
    Without this, one paper with five matching sentence windows would fill all the result slots."""
    seen, out = set(), []
    for c in chunks:
        if c['pmid'] not in seen:
            seen.add(c['pmid'])
            out.append(c)
    return out
