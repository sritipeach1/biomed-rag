"""Sanity-check the fetched corpus before building anything on it.

Run:  python -m biomedrag.ingest.stats
Copy the printed numbers into docs/data.md.
"""
import json
import random
from collections import Counter

import numpy as np

from biomedrag.config import CORPUS_PATH


def main(sample: int = 5, seed: int = 7):
    with open(CORPUS_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pmids = [r['pmid'] for r in rows]
    words = np.array([len(r['abstract'].split()) for r in rows])
    print(f'abstracts:            {len(rows)}')
    print(f'duplicate PMIDs:      {len(pmids) - len(set(pmids))}   (should be 0)')
    print(f'empty titles:         {sum(not r["title"].strip() for r in rows)}')
    print(f'empty abstracts:      {sum(not r["abstract"].strip() for r in rows)}')
    print(f'words per abstract:   mean {words.mean():.0f}, median {np.median(words):.0f}, '
          f'p95 {np.percentile(words, 95):.0f}, max {words.max()}')
    print(f'under 50 words:       {int((words < 50).sum())}   (dropped later in cleaning)')
    print('per year:', dict(sorted(Counter(r['year'] for r in rows).items())))
    print(f'\n--- {sample} random abstracts: read them yourself ---')
    random.seed(seed)
    for r in random.sample(rows, min(sample, len(rows))):
        print(f'\n[{r["pmid"]}] {r["title"]}\n{r["abstract"][:500]}...')


if __name__ == '__main__':
    main()
