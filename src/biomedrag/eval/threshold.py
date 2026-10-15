"""Choose the refusal threshold with evidence instead of a guess.

  python -m biomedrag.eval.threshold --split dev
  python -m biomedrag.eval.threshold --split test --final --use 3.2      (check the chosen number, once)

Every question goes through hybrid+rerank, and the best paper's reranker score is recorded. The pipeline
refuses when that score is below the threshold, so a higher threshold refuses more, including
questions it should have answered. The sweep shows the trade-off, and the script picks the cutoff that
meets your targets (default: refuse at least 80% of unanswerable questions, answer at least 85%
of answerable ones). If no cutoff meets both, it says so: that is a finding, not a failure.

Note what a score threshold cannot do: a question about glioma that the abstracts cannot answer
("cost at a private hospital") scores like a real question. Those are the near-miss questions; the
second refusal layer (the model's own INSUFFICIENT_EVIDENCE) has to catch them.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from biomedrag import config
from biomedrag.eval import metrics
from biomedrag.eval.questions import load_questions
from biomedrag.retrieve.retriever import Retriever


def collect_scores(retriever: Retriever, questions: list[dict]) -> list[dict]:
    """The best reranker score for every question (None if nothing was retrieved)."""
    rows = []
    for q in questions:
        results = retriever.retrieve(q['question'], mode='hybrid_rerank', k_final=config.K_FINAL,
                                     k_cand=config.K_CANDIDATES, k_rerank=config.K_RERANK)
        rows.append({'id': q['id'], 'source': q['source'], 'kind': q.get('kind'),
                     'answerable': q['answerable'], 'top_score': results[0][1] if results else None})
    return rows


def rate(rows: list[dict], threshold: float, answered: bool) -> float:
    """Share answered (or refused) at this threshold among the given rows."""
    scores = [-float('inf') if r['top_score'] is None else r['top_score'] for r in rows]
    return metrics.mean([1.0 if (s >= threshold) == answered else 0.0 for s in scores])


def breakdown(rows: list[dict], threshold: float) -> list[str]:
    lines = []
    for source in ('synthetic', 'handwritten'):
        group = [r for r in rows if r['answerable'] and r['source'] == source]
        if group:
            lines.append(f'  answered, {source:<12} questions: {rate(group, threshold, True):.0%}  (n={len(group)})')
    for kind in ('out_of_domain', 'near_miss'):
        group = [r for r in rows if not r['answerable'] and r['kind'] == kind]
        if group:
            lines.append(f'  refused,  {kind:<13} questions: {rate(group, threshold, False):.0%}  (n={len(group)})')
    return lines


def plot(sweep: list[dict], chosen: float | None, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    blue, orange = '#2a78d6', '#eb6834'
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=surface)
    xs = [r['threshold'] for r in sweep]
    ax.plot(xs, [r['answered'] for r in sweep], color=blue, linewidth=2, label='answerable questions still answered')
    ax.plot(xs, [r['refused'] for r in sweep], color=orange, linewidth=2, label='unanswerable questions refused')
    if chosen is not None:
        ax.axvline(chosen, color=ink2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(chosen, 0.03, f'  chosen: {chosen:.2f}', color=ink2, fontsize=8, va='bottom')
    ax.set_facecolor(surface)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel('refuse when the best reranker score is below', color=ink2, fontsize=9)
    ax.set_ylabel('share of questions', color=ink2, fontsize=9)
    ax.yaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink2, loc='lower left', bbox_to_anchor=(0, 1.0), ncols=2)
    fig.suptitle(title, fontsize=10, color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Calibrate the refusal threshold.')
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--final', action='store_true', help='required for --split test')
    ap.add_argument('--use', type=float, default=None,
                    help='report this threshold instead of choosing one (use it on test with the number chosen on dev)')
    ap.add_argument('--min-refuse', type=float, default=0.80)
    ap.add_argument('--min-answer', type=float, default=0.85)
    ap.add_argument('--strategy', default='abstract')
    args = ap.parse_args()
    if args.split == 'test' and not (args.final and args.use is not None):
        sys.exit('The test split is for reporting only: choose the threshold on dev, then run '
                 '--split test --final --use <that number>.')

    questions = load_questions(split=args.split)
    if not questions:
        sys.exit('No questions found. Build them first: python -m biomedrag.eval.build_set')
    print(f'loading the indexes and models ...\nscoring {len(questions)} questions ...')
    rows = collect_scores(Retriever.load(args.strategy), questions)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS / f'scores_{args.split}.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)

    answerable = [r['top_score'] for r in rows if r['answerable']]
    unanswerable = [r['top_score'] for r in rows if not r['answerable']]
    sweep = metrics.sweep_thresholds(answerable, unanswerable)
    with open(config.RESULTS / f'threshold_sweep_{args.split}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'answered', 'refused'])
        writer.writeheader()
        writer.writerows({k: f'{v:.4f}' for k, v in r.items()} for r in sweep)

    print(f'\n{len(answerable)} answerable and {len(unanswerable)} unanswerable questions ({args.split} split)')
    print(f'\n{"refuse below":>13}  {"answered":>9}  {"refused":>8}')
    step = max(1, len(sweep) // 14)
    for r in sweep[::step]:
        print(f'{r["threshold"]:>13.2f}  {r["answered"]:>9.0%}  {r["refused"]:>8.0%}')

    if args.use is not None:
        chosen, met = args.use, None
    else:
        best, met = metrics.choose_threshold(sweep, args.min_refuse, args.min_answer)
        chosen = best['threshold'] if best else None
    if chosen is None:
        sys.exit('nothing to choose from')
    print(f'\nthreshold {chosen:.2f}' + (' (given)' if met is None else
          ' meets both targets' if met else ' is the best compromise: NO threshold meets both targets'))
    print(f'  answerable questions answered: {rate([r for r in rows if r["answerable"]], chosen, True):.0%} '
          f'(target {args.min_answer:.0%})')
    print(f'  unanswerable questions refused: {rate([r for r in rows if not r["answerable"]], chosen, False):.0%} '
          f'(target {args.min_refuse:.0%})')
    print('\n'.join(breakdown(rows, chosen)))
    plot(sweep, chosen, config.ROOT / 'docs' / f'threshold_{args.split}.png',
         f'Refusal threshold sweep ({args.split} split)')
    print(f'\nchart: {config.ROOT / "docs" / f"threshold_{args.split}.png"}')
    if args.split == 'dev' and met is not None:
        print('\nWhen you are happy, put the number in .env as REFUSE_BELOW and note it in docs/results.md.')


if __name__ == '__main__':
    main()
