r"""Check the judge against YOUR reading of the papers.

  python -m biomedrag.eval.labels make  --split dev --run groq-openai-gpt-oss-120b-v2
  (open data\label_sheet.csv in Excel, fill the "your_label" column, save as CSV)
  python -m biomedrag.eval.labels score

`make` picks 30 judged sentences and writes a sheet with the question, the sentence, and the text of the
paper it cites. The judge's verdict is NOT in the sheet, so you cannot be swayed by it. For each row, read
the paper and type one letter in "your_label":
    s = supported   the paper states the claim or clearly implies it
    p = partial     part of the claim is backed, but a detail, number or conclusion is not
    n = not supported   the paper does not back it, contradicts it, or is about something else
`score` then compares your labels with the judge's: how often you agree, and where it is too lenient or too strict.

The sheet is mixed on purpose. If you first ran  judge.py --controls 8 , it also holds sentences whose citation was
swapped for a wrong paper (you are not told which): a judge that says "supported" to those cannot say no.
Because of this mix, the agreement is not the judge's accuracy on a typical answer.
"""
import argparse
import csv
import random
import sys
from collections import Counter

from biomedrag import config
from biomedrag.eval import metrics
from biomedrag.eval.answers import answers_path, load_done
from biomedrag.eval.judge import DEFAULT_JUDGE, VERDICTS, controls_path, judge_path

SHEET = config.DATA / 'label_sheet.csv'
KEY = config.DATA / 'label_key.csv'
LETTERS = {'s': 'supported', 'p': 'partial', 'n': 'unsupported'}


def pick(records: list[dict], n: int = 30, seed: int = 7, flagged_share: float = 1 / 3) -> list[dict]:
    """Choose n judged sentences: about a third the judge flagged (partial/unsupported), the rest supported.
    One sentence per answer first, so a single long answer cannot fill the sheet."""
    rng = random.Random(seed)
    items = [{'id': r['id'], 'question': r['question'], **s} for r in records for s in r['sentences']
             if s['verdict'] in VERDICTS]
    rng.shuffle(items)
    flagged = [i for i in items if i['verdict'] != 'supported']
    good = [i for i in items if i['verdict'] == 'supported']

    def spread(pool: list[dict], k: int) -> list[dict]:
        seen, first, rest = set(), [], []
        for item in pool:
            (rest if item['id'] in seen else first).append(item)
            seen.add(item['id'])
        return (first + rest)[:k]

    want_flagged = min(len(flagged), round(n * flagged_share))
    chosen = spread(flagged, want_flagged)
    chosen += spread(good, n - len(chosen))
    if len(chosen) < n:                                     # not enough supported ones: top up from the rest
        chosen += [i for i in flagged if i not in chosen][:n - len(chosen)]
    rng.shuffle(chosen)
    return chosen[:n]


def control_items(control_records: list[dict]) -> list[dict]:
    return [{'id': r['id'], 'question': r['question'], 'control': True, **s}
            for r in control_records for s in r['sentences'] if s['verdict'] in VERDICTS]


def make(records: list[dict], rows_by_id: dict[str, dict], n: int = 30, seed: int = 7,
         controls: list[dict] | None = None) -> int:
    """Write the blind sheet and the key. `controls` (judged wrong-paper sentences) are mixed in without a mark,
    so you cannot tell them apart; the key remembers which they are."""
    extra = control_items(controls or [])
    chosen = pick(records, max(n - len(extra), 0), seed) + extra
    random.Random(seed + 1).shuffle(chosen)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(SHEET, 'w', newline='', encoding='utf-8-sig') as f, open(KEY, 'w', newline='', encoding='utf-8') as k:
        sheet = csv.writer(f)
        key = csv.writer(k)
        sheet.writerow(['item', 'question', 'sentence', 'cited_paper_text', 'your_label (s / p / n)'])
        key.writerow(['item', 'answer_id', 'judge_verdict', 'sentence', 'control'])
        for i, item in enumerate(chosen, start=1):
            base = item['id'][2:] if item.get('control') else item['id']          # c_<id> -> <id>
            sources = {s['pmid']: s['text'] for s in rows_by_id[base]['sources']}
            papers = '\n\n'.join(f'[PMID:{p}] {sources.get(p, "(text not saved)")}' for p in item['pmids'])
            sheet.writerow([i, item['question'], item['text'], papers, ''])
            key.writerow([i, item['id'], item['verdict'], item['text'], 'yes' if item.get('control') else ''])
    return len(chosen)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float('nan')
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def read_labels(sheet_path=None) -> dict[int, str]:
    """{item number: 'supported' | 'partial' | 'unsupported'} for every row you filled in."""
    out: dict[int, str] = {}
    with open(sheet_path or SHEET, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 5 or not row[4].strip():
                continue
            letter = row[4].strip().lower()
            word = LETTERS.get(letter[0]) if letter else None
            if word is None:
                raise ValueError(f'item {row[0]}: label {row[4]!r} is not s, p or n')
            out[int(row[0])] = word
    return out


def score(labels: dict[int, str], key_path=None) -> list[str]:
    with open(key_path or KEY, newline='', encoding='utf-8') as f:
        key = {int(r['item']): r for r in csv.DictReader(f)}
    items = sorted(i for i in labels if i in key)
    mine = [labels[i] for i in items]
    judge = [key[i]['judge_verdict'] for i in items]
    agree = metrics.mean([1.0 if m == j else 0.0 for m, j in zip(mine, judge, strict=True)])
    binary = metrics.mean([1.0 if (m == 'supported') == (j == 'supported') else 0.0
                           for m, j in zip(mine, judge, strict=True)])
    lines = [f'items you labelled: {len(items)} of {len(key)}', '',
             f'agreement on all three verdicts : {agree:.0%}',
             f'agreement on supported vs not   : {binary:.0%}',
             (f"Cohen's kappa (3 verdicts)      : {cohen_kappa(mine, judge):.2f}   "
              '(0 = no better than chance, 1 = perfect; above 0.6 is usually called good)'), '',
             'rows = your label, columns = judge:', '',
             '| you \\ judge | ' + ' | '.join(VERDICTS) + ' |', '|---|' + '---|' * len(VERDICTS)]
    for v in VERDICTS:
        lines.append(f'| {v} | ' + ' | '.join(str(sum(1 for m, j in zip(mine, judge, strict=True) if m == v and j == w))
                                              for w in VERDICTS) + ' |')
    lenient = [i for i, m, j in zip(items, mine, judge, strict=True) if j == 'supported' and m != 'supported']
    strict = [i for i, m, j in zip(items, mine, judge, strict=True) if j != 'supported' and m == 'supported']
    ctrl = [i for i in items if key[i].get('control') == 'yes']
    if ctrl:
        mine_flag = sum(1 for i in ctrl if labels[i] != 'supported')
        judge_flag = sum(1 for i in ctrl if key[i]['judge_verdict'] != 'supported')
        lines += ['', f'wrong-paper control sentences (hidden in your sheet): {len(ctrl)}',
                  f'  you called {mine_flag} of {len(ctrl)} not fully supported; the judge flagged {judge_flag} of {len(ctrl)}']
    lines += ['', f'judge said supported, you said not   : {len(lenient)}  (judge too lenient)  items {lenient}',
              f'judge said not supported, you said yes : {len(strict)}  (judge too strict)   items {strict}']
    return lines


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Make a label sheet, or score your labels against the judge.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--split', choices=['dev', 'test'], default='dev')
    m.add_argument('--run', required=True)
    m.add_argument('--judge-model', default=DEFAULT_JUDGE)
    m.add_argument('--n', type=int, default=30)
    m.add_argument('--force', action='store_true', help='overwrite a sheet that already exists')
    sub.add_parser('score')
    args = ap.parse_args(argv)

    if args.cmd == 'make':
        if args.split == 'test':
            sys.exit('Label dev answers; the test split stays untouched.')
        if SHEET.exists() and not args.force:
            sys.exit(f'{SHEET} already exists (you may have labels in it). Use --force to replace it.')
        jp, ap_ = judge_path(args.split, args.run, args.judge_model), answers_path(args.split, args.run)
        if not jp.exists() or not ap_.exists():
            sys.exit('Run judge.py first for this run.')
        records = list(load_done(jp).values())
        cp = controls_path(args.split, args.run, args.judge_model)
        controls = list(load_done(cp).values()) if cp.exists() else []
        count = make(records, load_done(ap_), args.n, controls=controls)
        print(f'wrote {count} items to {SHEET}\nOpen it in Excel, fill the last column with s / p / n, save as CSV, '
              'then run:  python -m biomedrag.eval.labels score')
        return 0
    if not SHEET.exists() or not KEY.exists():
        sys.exit('No sheet yet. Run:  python -m biomedrag.eval.labels make --run <run name>')
    try:
        labels = read_labels()
    except ValueError as e:
        sys.exit(str(e))
    if not labels:
        sys.exit('The last column is empty. Fill it with s / p / n first.')
    text = score(labels)
    print('\n'.join(text))
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / 'judge_agreement.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
