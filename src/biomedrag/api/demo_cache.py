"""Saved answers to a few example questions.

The demo page offers these as buttons. They are answered from a file, so they work instantly, cost no
model calls, and never fail, even when the free model tier is busy.

Make or refresh the file (needs your model backend and the index):
    python -m biomedrag.api.demo_cache --file docs/demo_questions.txt --backend groq
Then READ the file's answers before you publish it.
"""
import argparse
import json
import sys
from pathlib import Path

from biomedrag import config
from biomedrag.api.schemas import present

CACHE_PATH = config.STATIC / 'demo_cache.json'


def key(question: str) -> str:
    return ' '.join(question.lower().split())


def load_cache(path: Path = CACHE_PATH) -> list[dict]:
    """[{'question': ..., 'response': {...}}, ...] in the order they were saved; [] if no file."""
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_questions(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith('#')]


def build(pipeline, questions: list[str], path: Path = CACHE_PATH, log=print) -> int:
    items = []
    for q in questions:
        r = present(pipeline.answer(q)).model_dump()
        items.append({'question': q, 'response': r})
        log(f'{r["status"]:22} {q}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding='utf-8')
    return len(items)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Save answers to the example questions for the demo page.')
    ap.add_argument('--file', default=str(config.ROOT / 'docs' / 'demo_questions.txt'))
    ap.add_argument('--backend', choices=['ollama', 'groq'], default=None)
    args = ap.parse_args(argv)
    from biomedrag.api.bootstrap import ensure_index
    from biomedrag.generate.pipeline import RAGPipeline
    ensure_index()
    pipeline = RAGPipeline.load(args.backend, 'abstract', refuse_below=config.REFUSE_BELOW,
                                prompt_version=config.PROMPT_VERSION)
    n = build(pipeline, read_questions(Path(args.file)))
    print(f'\nsaved {n} answers to {CACHE_PATH}\nOpen that file and read every answer before you publish it.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
