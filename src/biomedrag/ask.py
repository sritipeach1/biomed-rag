"""Ask a question and get a cited answer, from the command line.

  python -m biomedrag.ask "What is the role of IDH mutation in glioma prognosis?"
  python -m biomedrag.ask "your question" --backend groq
  python -m biomedrag.ask --file docs/manual_questions.txt --save data/ask_run1.jsonl
  python -m biomedrag.ask "your question" --refuse-below 3.5      (try a different refusal threshold)
"""
import argparse
import contextlib
import json
import sys
import textwrap

from biomedrag import config
from biomedrag.generate.llm import LLMError
from biomedrag.generate.pipeline import Answer, RAGPipeline


def show(a: Answer, raw: bool = False) -> None:
    score = f'{a.top_score:.2f}' if a.top_score is not None else 'n/a'
    print(f'\n=== {a.question}')
    print(f'status: {a.status}   best paper score: {score}   retries: {a.retries}   '
          f'search {a.retrieve_seconds:.1f}s, model {a.llm_seconds:.1f}s')
    print()
    for line in textwrap.wrap(a.text, width=100) or ['']:
        print(line)
    if a.report is not None and a.answered:
        r = a.report
        print(f'\ncitations: {len(r.valid)} papers cited, '
              f'{r.n_cited_sentences} of {r.n_sentences} sentences carry a citation')
    if raw:
        for i, attempt in enumerate(a.attempts, start=1):
            print(f'\n--- raw model reply, attempt {i} ---\n{attempt}\n--- end ---')
    print('\nPapers retrieved (* = cited in the answer):')
    for s in a.sources:
        title = s.title if len(s.title) <= 95 else s.title[:92] + '...'
        mark = '*' if s.cited else ' '
        sc = f'{s.score:6.2f}' if s.score is not None else '     -'
        print(f' {mark} PMID {s.pmid}  {sc}  {title}')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')     # Windows consoles and Greek letters

    ap = argparse.ArgumentParser(description='Answer a question from the indexed PubMed abstracts.')
    ap.add_argument('question', nargs='?')
    ap.add_argument('--file', help='a text file with one question per line')
    ap.add_argument('--backend', choices=['ollama', 'groq'], default=None,
                    help=f'which model writes the answer (default from .env: {config.LLM_BACKEND})')
    ap.add_argument('--strategy', default='abstract', help='which index folder to use')
    ap.add_argument('--refuse-below', type=float, default=config.REFUSE_BELOW,
                    help=f'refuse when the best paper scores below this (default {config.REFUSE_BELOW})')
    ap.add_argument('--show-raw', action='store_true',
                    help='also print exactly what the model replied (to debug refusals)')
    ap.add_argument('--save', help='also write every result to this .jsonl file')
    args = ap.parse_args()

    questions = [args.question] if args.question else []
    if args.file:
        with open(args.file, encoding='utf-8') as f:
            questions += [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    if not questions:
        ap.error('give a question, or --file with a list of questions')

    print('loading the indexes and models ...')
    try:
        pipeline = RAGPipeline.load(args.backend, args.strategy, refuse_below=args.refuse_below)
        print(f'model: {pipeline.llm.name} / {pipeline.llm.model}   refuse below: {args.refuse_below}')
        with (open(args.save, 'w', encoding='utf-8') if args.save else contextlib.nullcontext()) as out:
            for question in questions:
                answer = pipeline.answer(question)
                show(answer, args.show_raw)
                if out:
                    out.write(json.dumps(answer.to_dict(), ensure_ascii=False) + '\n')
                    out.flush()
    except LLMError as e:
        sys.exit(f'\nCould not get an answer: {e}')


if __name__ == '__main__':
    main()
