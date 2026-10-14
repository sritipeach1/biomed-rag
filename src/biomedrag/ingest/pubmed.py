"""Fetch PubMed abstracts for the corpus, one publication year at a time.

Run:  python -m biomedrag.ingest.pubmed
The script is resumable: rerun it after a failure and it skips PMIDs already saved.
"""
import json
import os
import time
from http.client import HTTPException

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

from biomedrag.config import CORPUS_PATH, DATA

load_dotenv()

BASE_QUERY = (
    '(glioma[Title/Abstract] OR glioblastoma[Title/Abstract]) '
    'AND hasabstract AND english[lang]'
)
YEARS = range(2018, 2027)


def configure_entrez() -> float:
    """Set the NCBI identity from the environment; return the pause between requests."""
    Entrez.email = os.environ['NCBI_EMAIL']
    Entrez.api_key = os.environ.get('NCBI_API_KEY') or None
    return 0.11 if Entrez.api_key else 0.34  # stay under the rate limit


def search_ids(year: int) -> list[str]:
    # PubMed search returns at most 10,000 records per query, hence one year at a time.
    term = f'{BASE_QUERY} AND {year}[dp]'
    handle = Entrez.esearch(db='pubmed', term=term, retmax=9999)
    ids = Entrez.read(handle)['IdList']
    handle.close()
    return ids


def parse(rec) -> dict | None:
    """Turn one PubmedArticle record into a flat dict, or None if it has no abstract."""
    cit = rec['MedlineCitation']
    art = cit['Article']
    parts = art.get('Abstract', {}).get('AbstractText', [])
    if not parts:
        return None
    pieces = []
    for p in parts:
        label = getattr(p, 'attributes', {}).get('Label')
        pieces.append(f'{label}: {p}' if label else str(p))
    pub = art['Journal']['JournalIssue'].get('PubDate', {})
    return {
        'pmid': str(cit['PMID']),
        'title': str(art.get('ArticleTitle', '')),
        'abstract': ' '.join(pieces),
        'journal': str(art['Journal'].get('Title', '')),
        'year': str(pub.get('Year', '')),
    }


def fetch_batch(ids: list[str], retries: int = 4):
    """Fetch one batch of records, retrying with a growing pause on network errors."""
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(db='pubmed', id=','.join(ids), retmode='xml')
            recs = Entrez.read(handle)['PubmedArticle']
            handle.close()
            return recs
        except (OSError, RuntimeError, ValueError, HTTPException) as exc:
            # network errors, HTTP 429/5xx, truncated or invalid XML
            wait = 2 ** (attempt + 1)
            print(f'  batch failed ({exc!r}); retrying in {wait}s')
            time.sleep(wait)
    return []


def fetch(ids: list[str], pause: float, batch: int = 200):
    for i in range(0, len(ids), batch):
        for rec in fetch_batch(ids[i:i + batch]):
            row = parse(rec)
            if row:
                yield row
        time.sleep(pause)


def load_seen() -> set[str]:
    if not CORPUS_PATH.exists():
        return set()
    with open(CORPUS_PATH) as f:
        return {json.loads(line)['pmid'] for line in f if line.strip()}


def main():
    pause = configure_entrez()
    DATA.mkdir(exist_ok=True)
    seen = load_seen()
    print(f'{len(seen)} abstracts already saved')
    with open(CORPUS_PATH, 'a') as f:
        for year in YEARS:
            ids = [i for i in search_ids(year) if i not in seen]
            print(year, len(ids), 'new records')
            time.sleep(pause)
            for row in tqdm(fetch(ids, pause), total=len(ids)):
                seen.add(row['pmid'])
                f.write(json.dumps(row) + '\n')
                f.flush()  # keep what we have if the run is interrupted
    print(f'done: {len(seen)} abstracts in {CORPUS_PATH}')


if __name__ == '__main__':
    main()
