"""Merge the three question groups into eval/questions.jsonl and check them.

  python -m biomedrag.eval.build_set
  python -m biomedrag.eval.build_set --skip-handwritten      (while you are still labelling)

Reads   eval/synthetic.jsonl      made by make_synthetic (and cleaned by you)
        eval/handwritten.txt      your questions, one per line:   h001 | question | 12345678, 23456789
        eval/unanswerable.jsonl   40 questions the abstracts cannot answer
Writes  eval/questions.jsonl
"""
import argparse
import sys
from pathlib import Path

from biomedrag import config
from biomedrag.eval.questions import corpus_pmids, read_jsonl, summary, validate, write_jsonl

SYNTHETIC = config.EVAL / 'synthetic.jsonl'
HANDWRITTEN = config.EVAL / 'handwritten.txt'
UNANSWERABLE = config.EVAL / 'unanswerable.jsonl'


def parse_handwritten(path: Path) -> tuple[list[dict], list[str]]:
    """Read lines like  h001 | question | 123, 456.  Returns (rows, ids of lines that have no PMID yet)."""
    rows, unlabelled = [], []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) != 3 or not parts[0] or not parts[1]:
            raise ValueError(f'{path.name} line {number}: expected  id | question | pmid, pmid   got: {line!r}')
        qid, question, pmid_text = parts
        pmids = [p for p in (x.strip() for x in pmid_text.replace(';', ',').split(',')) if p]
        bad = [p for p in pmids if not p.isdigit()]
        if bad:
            raise ValueError(f'{path.name} line {number}: PMIDs must be numbers, got {bad}')
        if not pmids:
            unlabelled.append(qid)
        rows.append({'id': qid, 'question': question, 'answerable': True, 'gold_pmids': pmids,
                     'source': 'handwritten', 'kind': None})
    return rows, unlabelled


def build(skip_handwritten: bool = False, check_index: bool = True,
          out: Path | None = None) -> tuple[list[dict], list[str]]:
    """Return (rows, problems). Writes the file only when there are no problems."""
    rows: list[dict] = []
    problems: list[str] = []
    if SYNTHETIC.exists():
        rows += read_jsonl(SYNTHETIC)
    else:
        problems.append(f'{SYNTHETIC.name} not found: run  python -m biomedrag.eval.make_synthetic')
    if not skip_handwritten:
        if HANDWRITTEN.exists():
            hand, unlabelled = parse_handwritten(HANDWRITTEN)
            if unlabelled:
                problems.append(f'{len(unlabelled)} hand-written question(s) have no PMID yet '
                                f'({", ".join(unlabelled[:5])}{" ..." if len(unlabelled) > 5 else ""}). '
                                f'Fill them in, or use --skip-handwritten for now.')
            rows += hand
        else:
            problems.append(f'{HANDWRITTEN.name} not found (or use --skip-handwritten)')
    rows += read_jsonl(UNANSWERABLE)
    pmids = corpus_pmids() if check_index else None
    problems += validate(rows, pmids)
    if not problems:
        write_jsonl(rows, out or config.QUESTIONS_PATH)
    return rows, problems


def main():
    ap = argparse.ArgumentParser(description='Build eval/questions.jsonl.')
    ap.add_argument('--skip-handwritten', action='store_true')
    args = ap.parse_args()
    rows, problems = build(args.skip_handwritten)
    if problems:
        print('Not written. Fix these first:')
        print('\n'.join(f'  - {p}' for p in problems))
        sys.exit(1)
    print(f'wrote {len(rows)} questions to {config.QUESTIONS_PATH}\n')
    print(summary(rows))


if __name__ == '__main__':
    main()
