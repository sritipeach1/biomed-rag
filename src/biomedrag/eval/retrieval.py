"""Measure how well the search finds the right paper, and run the ablations.

  python -m biomedrag.eval.retrieval --split dev
  python -m biomedrag.eval.retrieval --split dev --failures 5
  python -m biomedrag.eval.retrieval --split dev --strategy sentences       (after building that index)
  python -m biomedrag.eval.retrieval --split test --final                   (once, at the very end)

For every answerable question, the search returns its top 10 papers. Metrics:
  recall@k   share of questions whose gold paper is in the top k          (k = 1, 5, 10)
  MRR@10     mean of 1 / (rank of the gold paper); 0 if it is not in the top 10
  p50/p95    milliseconds per question (median and slow tail)
The ablation grid: bm25, dense, hybrid, and hybrid+rerank with 10, 20 or 50 candidates re-scored.
Every run is appended to eval/results/retrieval.csv. Synthetic and hand-written questions are always
reported separately: synthetic ones share words with their source abstract, which flatters BM25.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from biomedrag import config
from biomedrag.eval import metrics
from biomedrag.eval.questions import load_questions
from biomedrag.retrieve.retriever import Retriever

K_FINAL = 10
GRID = [('bm25', None), ('dense', None), ('hybrid', None),
        ('hybrid_rerank', 10), ('hybrid_rerank', 20), ('hybrid_rerank', 50)]
CSV_COLUMNS = ['timestamp', 'split', 'source', 'strategy', 'mode', 'k_rerank', 'n', 'recall@1', 'recall@5',
               'recall@10', 'mrr@10', 'recall@5_lo', 'recall@5_hi', 'mrr_lo', 'mrr_hi', 'p50_ms', 'p95_ms']


def label(mode: str, k_rerank: int | None) -> str:
    return mode if k_rerank is None else f'{mode}@{k_rerank}'


def run_config(retriever: Retriever, questions: list[dict], mode: str, k_rerank: int | None) -> list[dict]:
    """Search every question once and record where its gold paper ended up."""
    rows = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.retrieve(q['question'], mode=mode, k_final=K_FINAL,
                                     k_cand=config.K_CANDIDATES, k_rerank=k_rerank or config.K_RERANK)
        ms = (time.perf_counter() - start) * 1000
        pmids = [c['pmid'] for c, _ in results]
        rows.append({'id': q['id'], 'source': q['source'], 'rank': metrics.first_gold_rank(pmids, q['gold_pmids']),
                     'ms': ms, 'top_pmids': pmids[:5]})
    return rows


def summarize(rows: list[dict]) -> dict:
    ranks = [r['rank'] for r in rows]
    r5, rr = metrics.hits(ranks, 5), metrics.reciprocal_ranks(ranks, K_FINAL)
    ms = [r['ms'] for r in rows]
    r5_lo, r5_hi = metrics.bootstrap_ci(r5)
    rr_lo, rr_hi = metrics.bootstrap_ci(rr)
    return {'n': len(rows), 'recall@1': metrics.mean(metrics.hits(ranks, 1)), 'recall@5': metrics.mean(r5),
            'recall@10': metrics.mean(metrics.hits(ranks, 10)), 'mrr@10': metrics.mean(rr),
            'recall@5_lo': r5_lo, 'recall@5_hi': r5_hi, 'mrr_lo': rr_lo, 'mrr_hi': rr_hi,
            'p50_ms': metrics.percentile(ms, 50), 'p95_ms': metrics.percentile(ms, 95)}


def by_source(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r['source'], []).append(r)
    return groups


def append_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if fresh:
            writer.writeheader()
        for rec in records:
            writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in rec.items()})


def markdown_table(summaries: dict[str, dict], source: str) -> str:
    head = '| system | n | recall@1 | recall@5 | recall@10 | MRR@10 | p50 ms | p95 ms |\n|---|---|---|---|---|---|---|---|'
    lines = [f'**{source}**', '', head]
    for name, s in summaries.items():
        lines.append(f'| {name} | {s["n"]} | {s["recall@1"]:.3f} | {s["recall@5"]:.3f} '
                     f'[{s["recall@5_lo"]:.2f}, {s["recall@5_hi"]:.2f}] | {s["recall@10"]:.3f} | '
                     f'{s["mrr@10"]:.3f} [{s["mrr_lo"]:.2f}, {s["mrr_hi"]:.2f}] | '
                     f'{s["p50_ms"]:.0f} | {s["p95_ms"]:.0f} |')
    return '\n'.join(lines)


def paired_lines(results: dict[str, list[dict]], source: str, a: str, b: str) -> list[str]:
    """How much better is system a than system b, on the same questions, with an interval."""
    ra = {r['id']: r for r in results[a] if r['source'] == source}
    rb = {r['id']: r for r in results[b] if r['source'] == source}
    ids = sorted(ra.keys() & rb.keys())
    if not ids:
        return []
    out = []
    for name, fn in (('recall@5', lambda rows: metrics.hits([r['rank'] for r in rows], 5)),
                     ('MRR@10', lambda rows: metrics.reciprocal_ranks([r['rank'] for r in rows], K_FINAL))):
        diff, lo, hi = metrics.paired_bootstrap(fn([ra[i] for i in ids]), fn([rb[i] for i in ids]))
        verdict = 'a real difference' if lo > 0 or hi < 0 else 'could be chance'
        out.append(f'  {a} minus {b}, {name}: {diff:+.3f}  (95% interval {lo:+.3f} to {hi:+.3f}: {verdict})')
    return out


def show_failures(results: list[dict], questions: dict[str, dict], titles: dict[str, str], n: int) -> None:
    bad = [r for r in results if r['rank'] is None or r['rank'] > 5][:n]
    print(f'\nQuestions where the gold paper is not in the top 5 ({len(bad)} shown):')
    for r in bad:
        q = questions[r['id']]
        print(f'\n- {q["id"]}: {q["question"]}')
        print(f'  gold rank: {r["rank"] or "not in top 10"}')
        for pmid in q['gold_pmids']:
            print(f'  gold    PMID {pmid}: {titles.get(pmid, "?")[:100]}')
        for pmid in r['top_pmids'][:3]:
            print(f'  found   PMID {pmid}: {titles.get(pmid, "?")[:100]}')


def plot(summaries_by_source: dict[str, dict[str, dict]], path: Path, title: str) -> None:
    """Grouped bars: recall@5 and MRR@10 for every system, one panel per question source."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    blue, orange = '#2a78d6', '#eb6834'
    panels = list(summaries_by_source.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 1.2, 4.2), squeeze=False,
                             sharey=True, facecolor=surface)
    for ax, (source, summ) in zip(axes[0], panels, strict=True):
        names = list(summ)
        xs = range(len(names))
        w = 0.38
        for offset, key, color, name in ((-w / 2, 'recall@5', blue, 'recall@5'), (w / 2, 'mrr@10', orange, 'MRR@10')):
            vals = [summ[n][key] for n in names]
            bars = ax.bar([x + offset for x in xs], vals, w, color=color, label=name,
                          edgecolor=surface, linewidth=1.5)
            for bar, v in zip(bars, vals, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=7.5, color=ink2)
        ax.set_facecolor(surface)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n.replace('hybrid_rerank', 'hybrid+\nrerank') for n in names], fontsize=8, color=ink2)
        ax.set_ylim(0, 1.08)
        ax.set_title(f'{source} questions (n={next(iter(summ.values()))["n"]})', fontsize=10, color=ink, loc='left')
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(grid)
        ax.tick_params(axis='y', colors=ink2, labelsize=8, length=0)
        ax.tick_params(axis='x', length=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, y=0.985, ha='left')
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=ink2, loc='upper left',
               bbox_to_anchor=(0.005, 0.935), ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Retrieval evaluation and ablations.')
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--final', action='store_true', help='required for --split test: you are reporting, not tuning')
    ap.add_argument('--strategy', default='abstract')
    ap.add_argument('--source', choices=['all', 'synthetic', 'handwritten'], default='all')
    ap.add_argument('--failures', type=int, default=0, help='also show N questions the best system missed')
    ap.add_argument('--no-chart', action='store_true')
    args = ap.parse_args()
    if args.split == 'test' and not args.final:
        sys.exit('The test split is for the final numbers only. Tune on --split dev. '
                 'If you really are done tuning, add --final.')

    questions = [q for q in load_questions(split=args.split, answerable=True)
                 if args.source in ('all', q['source'])]
    if not questions:
        sys.exit('No answerable questions in this split. Build them first: python -m biomedrag.eval.build_set')
    print('loading the indexes and models ...')
    retriever = Retriever.load(args.strategy)
    titles = {c['pmid']: c['title'] for c in retriever.chunks}
    for mode, kr in GRID:                                   # warm-up, so the first timing is fair
        retriever.retrieve('glioma', mode=mode, k_final=K_FINAL, k_rerank=kr or config.K_RERANK)

    results: dict[str, list[dict]] = {}
    for mode, kr in GRID:
        name = label(mode, kr)
        print(f'running {name} on {len(questions)} questions ...')
        results[name] = run_config(retriever, questions, mode, kr)

    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    records, summaries_by_source = [], {}
    for source in sorted({q['source'] for q in questions}):
        summaries = {}
        for (mode, kr) in GRID:
            name = label(mode, kr)
            rows = [r for r in results[name] if r['source'] == source]
            summaries[name] = s = summarize(rows)
            records.append({'timestamp': stamp, 'split': args.split, 'source': source, 'strategy': args.strategy,
                            'mode': mode, 'k_rerank': kr or '', **s})
        summaries_by_source[source] = summaries
        print('\n' + markdown_table(summaries, f'{source} questions, {args.split} split, chunking: {args.strategy}'))
        for line in paired_lines(results, source, 'hybrid_rerank@20', 'bm25'):
            print(line)
    append_csv(config.RESULTS / 'retrieval.csv', records)

    perq = config.RESULTS / f'retrieval_{args.strategy}_{args.split}_perquestion.jsonl'
    with open(perq, 'w', encoding='utf-8') as f:
        for name, rows in results.items():
            f.writelines(json.dumps({'system': name, **r}) + '\n' for r in rows)
    print(f'\nadded {len(records)} rows to {config.RESULTS / "retrieval.csv"}')
    if args.failures:
        show_failures(results['hybrid_rerank@20'], {q['id']: q for q in questions}, titles, args.failures)
    if not args.no_chart:
        chart = config.ROOT / 'docs' / f'retrieval_{args.strategy}_{args.split}.png'
        plot(summaries_by_source, chart, f'Retrieval ablation ({args.split} split, chunking: {args.strategy})')
        print(f'chart: {chart}')


if __name__ == '__main__':
    main()
