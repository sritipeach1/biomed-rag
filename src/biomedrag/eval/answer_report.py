"""Compare saved runs (made by answers.py) at one or more refusal thresholds.

  python -m biomedrag.eval.answer_report --split dev --runs groq-openai-gpt-oss-120b-v1 groq-openai-gpt-oss-120b-v2
  python -m biomedrag.eval.answer_report --split dev --runs <run> --thresholds 2.0 5.0 5.47

The score threshold is applied here, afterwards: a question whose best paper scores below it counts as
refused, exactly as the live pipeline would have refused it. The first run listed is the baseline.

What the numbers mean
- decision correct: answerable question answered, or unanswerable question refused. This is the headline.
- leaked: unanswerable questions that still got an answer (the dangerous kind).
- citation coverage: share of answer sentences that carry a citation.
- gold cited: the paper the question was written from is among the papers the answer cites.
  This is "did it use the right paper", not "is the sentence true" (that is the faithfulness check).
"""
import argparse
import csv
import sys

from biomedrag import config
from biomedrag.eval import metrics
from biomedrag.eval.answers import answers_path, load_done


def effective_status(row: dict, threshold: float | None) -> str:
    """The status the live pipeline would have produced at this threshold."""
    if row['status'] == 'refused_no_results' or row['top_score'] is None:
        return 'refused_no_results'
    if threshold is not None and row['top_score'] < threshold:
        return 'refused_low_score'
    return row['status']


def is_answered(row: dict, threshold: float | None) -> bool:
    return effective_status(row, threshold) == 'answered'


def decision_correct(row: dict, threshold: float | None) -> float:
    return 1.0 if is_answered(row, threshold) == row['answerable'] else 0.0


def coverage(row: dict) -> float:
    r = row.get('report')
    return r['n_cited_sentences'] / r['n_sentences'] if r and r['n_sentences'] else 0.0


def gold_cited(row: dict) -> float:
    gold = set(row['gold_pmids'])
    return 1.0 if gold & set((row.get('report') or {}).get('valid', [])) else 0.0


def gold_retrieved(row: dict) -> float:
    gold = set(row['gold_pmids'])
    return 1.0 if gold & {s['pmid'] for s in row['sources']} else 0.0


def evaluate(rows: list[dict], threshold: float | None) -> dict:
    """All the numbers for one run at one threshold. Lists are per-question values (for intervals)."""
    ans = [r for r in rows if r['answerable']]
    un = [r for r in rows if not r['answerable']]
    answered_ans = [r for r in ans if is_answered(r, threshold)]
    seconds = [r['retrieve_seconds'] + r['llm_seconds'] for r in rows]
    return {
        'n_answerable': len(ans), 'n_unanswerable': len(un),
        'decision_correct': [decision_correct(r, threshold) for r in rows],
        'answerable_answered': [1.0 if is_answered(r, threshold) else 0.0 for r in ans],
        'unanswerable_refused': [0.0 if is_answered(r, threshold) else 1.0 for r in un],
        'leaked': sorted(r['id'] for r in un if is_answered(r, threshold)),
        'wrongly_refused': {r['id']: effective_status(r, threshold) for r in ans if not is_answered(r, threshold)},
        'refused_by_model_or_citations': sum(1 for r in ans
                                             if effective_status(r, threshold) in ('refused_by_model', 'refused_bad_citations')),
        'coverage': [coverage(r) for r in answered_ans],
        'fully_cited': [1.0 if coverage(r) == 1.0 else 0.0 for r in answered_ans],
        'gold_cited': [gold_cited(r) for r in answered_ans if r['gold_pmids']],
        'gold_retrieved': [gold_retrieved(r) for r in ans if r['gold_pmids']],
        'retried': [1.0 if r['retries'] else 0.0 for r in rows if r['status'] != 'refused_no_results'],
        'p50': metrics.percentile(seconds, 50), 'p95': metrics.percentile(seconds, 95),
    }


def fmt_rate(values: list[float]) -> str:
    if not values:
        return 'n/a'
    lo, hi = metrics.bootstrap_ci(values)
    return f'{metrics.mean(values):.0%} [{lo:.0%}, {hi:.0%}] (n={len(values)})'


METRIC_ROWS = [
    ('decision correct (all questions)', 'decision_correct'),
    ('answerable: answered', 'answerable_answered'),
    ('unanswerable: refused', 'unanswerable_refused'),
    ('answers: sentences cited (mean)', 'coverage'),
    ('answers: every sentence cited', 'fully_cited'),
    ('answers: gold paper cited', 'gold_cited'),
    ('search: gold paper in the 5 given to the model', 'gold_retrieved'),
    ('questions needing a retry', 'retried'),
]


def table(results: dict[str, dict], threshold: float | None) -> list[str]:
    names = list(results)
    label = 'no threshold' if threshold is None else f'threshold {threshold:g}'
    lines = [f'**{label}**', '', '| metric | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for title, key in METRIC_ROWS:
        cells = []
        for name in names:
            v = results[name][key]
            cells.append(f'{metrics.mean(v):.0%} (n={len(v)})' if key == 'coverage' and v else fmt_rate(v))
        lines.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    lines.append('| leaked unanswerable questions | ' +
                 ' | '.join(f'{len(results[n]["leaked"])}: {", ".join(results[n]["leaked"]) or "-"}' for n in names) + ' |')
    lines.append('| answerable but refused by the model | ' +
                 ' | '.join(str(results[n]['refused_by_model_or_citations']) for n in names) + ' |')
    lines.append('| time per question p50 / p95 | ' +
                 ' | '.join(f'{results[n]["p50"]:.1f}s / {results[n]["p95"]:.1f}s' for n in names) + ' |')
    return lines


def compare_first_two(rows_by_run: dict[str, list[dict]], threshold: float | None) -> list[str]:
    names = list(rows_by_run)
    if len(names) < 2:
        return []
    a, b = names[1], names[0]
    ra, rb = rows_by_run[a], rows_by_run[b]
    diff, lo, hi = metrics.paired_bootstrap([decision_correct(r, threshold) for r in ra],
                                            [decision_correct(r, threshold) for r in rb])
    verdict = 'unlikely to be chance' if (lo > 0 or hi < 0) else 'could be chance'
    lines = [f'  decision correct, {a} minus {b}: {diff:+.3f}  (95% interval {lo:+.3f} to {hi:+.3f}: {verdict})']
    changed = [(x['id'], x['answerable'], effective_status(y, threshold), effective_status(x, threshold))
               for x, y in zip(ra, rb, strict=True)
               if is_answered(x, threshold) != is_answered(y, threshold)]
    if changed:
        lines.append(f'  questions where {a} decided differently from {b}:')
        for qid, answerable, old, new in changed:
            kind = 'answerable  ' if answerable else 'unanswerable'
            lines.append(f'    {qid:<12} {kind}  {old} -> {new}')
    return lines


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Compare saved answer runs.')
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--final', action='store_true', help='required for --split test')
    ap.add_argument('--runs', nargs='+', required=True, help='run names printed by answers.py (first = baseline)')
    ap.add_argument('--thresholds', nargs='*', type=float, default=[2.0, 5.0, 5.47])
    args = ap.parse_args()
    if args.split == 'test' and not args.final:
        sys.exit('The test split is for reporting only; use --final once.')

    loaded = {}
    for run in args.runs:
        path = answers_path(args.split, run)
        if not path.exists():
            sys.exit(f'No saved answers at {path}. Run answers.py first.')
        loaded[run] = load_done(path)
    common = set.intersection(*(set(r) for r in loaded.values()))
    for run, rows in loaded.items():
        if len(rows) != len(common):
            print(f'note: {run} has {len(rows)} questions, only the {len(common)} that every run has are compared')
    order = sorted(common)
    rows_by_run = {run: [rows[i] for i in order] for run, rows in loaded.items()}
    if not order:
        sys.exit('The runs have no questions in common.')

    out_rows = []
    for t in args.thresholds:
        results = {run: evaluate(rows, t) for run, rows in rows_by_run.items()}
        print('\n' + '\n'.join(table(results, t)))
        for line in compare_first_two(rows_by_run, t):
            print(line)
        for run, res in results.items():
            for title, key in METRIC_ROWS:
                v = res[key]
                lo, hi = metrics.bootstrap_ci(v)
                out_rows.append({'run': run, 'threshold': t, 'metric': title, 'value': f'{metrics.mean(v):.4f}',
                                 'ci_low': f'{lo:.4f}', 'ci_high': f'{hi:.4f}', 'n': len(v)})
            out_rows.append({'run': run, 'threshold': t, 'metric': 'leaked unanswerable',
                             'value': len(res['leaked']), 'ci_low': '', 'ci_high': '', 'n': res['n_unanswerable']})
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS / f'answers_summary_{args.split}.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'threshold', 'metric', 'value', 'ci_low', 'ci_high', 'n'])
        w.writeheader()
        w.writerows(out_rows)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
