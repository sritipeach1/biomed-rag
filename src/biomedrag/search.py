"""Try the search yourself, from the command line.

  python -m biomedrag.search "What is the role of IDH mutation in glioma prognosis?"
  python -m biomedrag.search "your question" --compare          (all four modes side by side)
  python -m biomedrag.search --file docs/manual_questions.txt --compare
  python -m biomedrag.search "your question" --strategy sentences --show-text
"""
import argparse
import sys

from biomedrag import config
from biomedrag.retrieve.retriever import Retriever


def show(results, show_text: bool = False) -> None:
    if not results:
        print('   (no results)')
    for rank, (chunk, score) in enumerate(results, start=1):
        shown = f'  score {score:6.2f}' if score is not None else ''
        title = chunk['title'] if len(chunk['title']) <= 105 else chunk['title'][:102] + '...'
        print(f'{rank}. PMID {chunk["pmid"]}{shown}  {title}')
        if show_text:
            print(f'     {chunk["text"][:300]}...')


def run_question(retriever: Retriever, question: str, modes, k: int, show_text: bool) -> None:
    print(f'\n=== {question}')
    for mode in modes:
        print(f'\n[{mode}]')
        show(retriever.retrieve(question, mode=mode, k_final=k), show_text)


def main():
    # Titles contain symbols (Greek letters, dashes) that the Windows console cannot always print.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser(description='Search the indexed PubMed abstracts.')
    ap.add_argument('question', nargs='?', help='the question to search for')
    ap.add_argument('--file', help='a text file with one question per line')
    ap.add_argument('--strategy', default='abstract', help='which index folder to use')
    ap.add_argument('--mode', default='hybrid_rerank', choices=config.MODES)
    ap.add_argument('--compare', action='store_true', help='show all four modes for each question')
    ap.add_argument('-k', type=int, default=config.K_FINAL, help='how many papers to show')
    ap.add_argument('--show-text', action='store_true', help='also show the start of each passage')
    args = ap.parse_args()

    questions = []
    if args.question:
        questions.append(args.question)
    if args.file:
        with open(args.file, encoding='utf-8') as f:
            questions += [line.strip() for line in f if line.strip() and not line.startswith('#')]
    if not questions:
        ap.error('give a question, or --file with a list of questions')

    print('loading the indexes and models ...')
    retriever = Retriever.load(args.strategy)
    modes = config.MODES if args.compare else (args.mode,)
    for question in questions:
        run_question(retriever, question, modes, args.k, args.show_text)


if __name__ == '__main__':
    main()
