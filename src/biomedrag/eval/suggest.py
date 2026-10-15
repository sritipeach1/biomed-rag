"""Help you find the relevant papers for your hand-written questions.

  python -m biomedrag.eval.suggest

Reads   eval/handwritten.txt                      (questions; the PMID column may still be empty)
Writes  data/handwritten_candidates.md            open it in VS Code

For each question it lists the top papers from three different searches (keyword, meaning, and
hybrid+rerank), merged. Read the abstracts, decide which papers really answer the question, and
type their PMIDs into eval/handwritten.txt. Papers that ALL searches miss can still be relevant:
also try the question on pubmed.ncbi.nlm.nih.gov, so the gold list is not shaped only by our own system.
"""
import sys

from biomedrag import config
from biomedrag.eval.build_set import HANDWRITTEN, parse_handwritten
from biomedrag.retrieve.retriever import Retriever

POOL_MODES = ('bm25', 'dense', 'hybrid_rerank')
OUT = config.DATA / 'handwritten_candidates.md'


def pool(retriever: Retriever, question: str, per_mode: int = 10) -> list[dict]:
    """Merge the top results of several searches. Papers found by more searches come first."""
    found: dict[str, dict] = {}
    for mode in POOL_MODES:
        for rank, (chunk, _) in enumerate(retriever.retrieve(question, mode=mode, k_final=per_mode), 1):
            entry = found.setdefault(chunk['pmid'], {'chunk': chunk, 'modes': [], 'best_rank': rank})
            entry['modes'].append(mode)
            entry['best_rank'] = min(entry['best_rank'], rank)
    return sorted(found.values(), key=lambda e: (-len(e['modes']), e['best_rank']))


def render(rows: list[dict], retriever: Retriever) -> str:
    out = ['# Candidate papers for your hand-written questions', '',
           'Tick the ones that really answer the question, then put their PMIDs in eval/handwritten.txt.', '']
    for row in rows:
        out += [f'## {row["id"]}  {row["question"]}', '',
                f'Currently labelled: {", ".join(row["gold_pmids"]) or "(none yet)"}', '']
        for e in pool(retriever, row['question']):
            c = e['chunk']
            out += [f'- [ ] **PMID {c["pmid"]}** ({c["year"]}) found by: {", ".join(e["modes"])}',
                    f'  {c["text"][:420]}...',
                    f'  https://pubmed.ncbi.nlm.nih.gov/{c["pmid"]}/', '']
    return '\n'.join(out)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    rows, _ = parse_handwritten(HANDWRITTEN)
    print('loading the indexes and models ...')
    retriever = Retriever.load('abstract')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows, retriever), encoding='utf-8')
    print(f'wrote {OUT}  ({len(rows)} questions). Open it in VS Code.')


if __name__ == '__main__':
    main()
