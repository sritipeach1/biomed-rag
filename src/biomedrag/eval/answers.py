"""Run the full pipeline (search + model) over the evaluation questions and save every answer.

  python -m biomedrag.eval.answers --split dev --backend groq --prompt v1
  python -m biomedrag.eval.answers --split dev --backend groq --prompt v2
  python -m biomedrag.eval.answers --split dev --backend ollama --prompt v2     (local model, for the comparison)
  python -m biomedrag.eval.answers --split dev --backend groq --prompt v1 --limit 5    (try 5 first)
  python -m biomedrag.eval.answers --split dev --backend groq --prompt v2 --source unanswerable handwritten
                                                  (spend a small daily quota where it matters most)

Why it is built this way
- The score threshold is switched OFF here (the model is asked every time) and applied later, in
  answer_report.py. That way ONE run can be evaluated at many thresholds without asking the model again.
- Each finished question is appended to data/answers/answers_<split>_<model>_<prompt>.jsonl straight away.
  If it stops (rate limit, laptop closed), run the same command again: finished questions are skipped.
- The files stay in data/ (not committed): they hold the retrieved abstract texts. The small summary
  tables that answer_report.py writes to eval/results/ are what get committed.
- The test split needs --final: it is for one last report, not for tuning.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from biomedrag import config
from biomedrag.eval.questions import load_questions
from biomedrag.generate.llm import LLMError
from biomedrag.generate.pipeline import RAGPipeline

ANSWERS = config.DATA / 'answers'
DAILY_LIMIT = ('tokens per day', 'TPD')      # Groq's free daily allowance: waiting a minute will not help


def run_tag(llm, prompt: str) -> str:
    """A file-name-safe label such as groq-openai-gpt-oss-120b-v1."""
    raw = f'{llm.name}-{llm.model}-{prompt}'
    return re.sub(r'[^A-Za-z0-9.]+', '-', raw).strip('-')


def answers_path(split: str, tag: str) -> Path:
    return ANSWERS / f'answers_{split}_{tag}.jsonl'


def load_done(path: Path) -> dict[str, dict]:
    """Rows already saved, by question id. A half-written last line (a crash) is ignored."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row['id']] = row
    return done


def run(pipeline: RAGPipeline, questions: list[dict], path: Path, sleep: float = 0.0,
        max_errors: int = 3, log=print) -> tuple[int, int, int]:
    """Answer every question that is not in `path` yet. Returns (new, skipped, failed).
    Stops early after `max_errors` model failures in a row (probably a rate limit or a wrong key)."""
    done = load_done(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = failed = in_a_row = 0
    todo = [q for q in questions if q['id'] not in done]
    with open(path, 'a', encoding='utf-8') as out:
        for i, q in enumerate(todo, start=1):
            try:
                answer = pipeline.answer(q['question'])
            except LLMError as e:
                failed += 1
                in_a_row += 1
                log(f'  [{i}/{len(todo)}] {q["id"]}: model error, will be retried next run: {e}')
                if any(w in str(e) for w in DAILY_LIMIT):
                    log('  the daily token allowance is used up. Run the same command again later '
                        '(it takes up to 24 hours to free up completely; finished questions are kept).')
                    break
                if in_a_row >= max_errors:
                    log(f'  stopping after {max_errors} errors in a row. Run the same command again later.')
                    break
                continue
            in_a_row = 0
            row = {'id': q['id'], 'source': q['source'], 'kind': q.get('kind'),
                   'answerable': q['answerable'], 'gold_pmids': q['gold_pmids'], **answer.to_dict()}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            new += 1
            score = f'{answer.top_score:.2f}' if answer.top_score is not None else 'n/a'
            log(f'  [{i}/{len(todo)}] {q["id"]}: {answer.status}  (best score {score}, '
                f'model {answer.llm_seconds:.1f}s)')
            if sleep:
                time.sleep(sleep)
    return new, len(questions) - len(todo), failed


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Answer the evaluation questions and save the results.')
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--final', action='store_true', help='required for --split test')
    ap.add_argument('--backend', choices=['ollama', 'groq'], default=None)
    ap.add_argument('--prompt', choices=['v1', 'v2', 'v3'], required=True, help='which instructions the model gets')
    ap.add_argument('--strategy', default='abstract')
    ap.add_argument('--source', nargs='+', choices=['synthetic', 'handwritten', 'unanswerable'], default=None,
                    help='only these kinds of question (the report compares runs on the questions they share)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N questions (to try it out)')
    ap.add_argument('--sleep', type=float, default=1.0, help='seconds to wait between questions (rate limits)')
    args = ap.parse_args()
    if args.split == 'test' and not args.final:
        sys.exit('The test split is for reporting only. Tune on dev; run the test split once with --final.')

    questions = load_questions(split=args.split)
    if not questions:
        sys.exit('No questions found. Build them first: python -m biomedrag.eval.build_set')
    if args.source:
        questions = [q for q in questions if q['source'] in args.source]
    if args.limit:
        questions = questions[:args.limit]

    print('loading the indexes and models ...')
    try:
        pipeline = RAGPipeline.load(args.backend, args.strategy, refuse_below=float('-inf'),
                                    prompt_version=args.prompt)
    except LLMError as e:
        sys.exit(f'Could not start: {e}')
    tag = run_tag(pipeline.llm, args.prompt)
    path = answers_path(args.split, tag)
    print(f'run: {tag}   questions: {len(questions)}   file: {path}')
    new, skipped, failed = run(pipeline, questions, path, sleep=args.sleep)
    print(f'\ndone: {new} new, {skipped} already saved, {failed} failed.')
    if failed:
        print('Run the same command again to retry the failed ones.')
    print('Next:  python -m biomedrag.eval.answer_report --split', args.split, '--runs', tag)


if __name__ == '__main__':
    main()
