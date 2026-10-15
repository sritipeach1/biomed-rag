r"""Delete chosen questions from eval/synthetic.jsonl without editing JSON by hand.

  python -m biomedrag.eval.prune --file eval\prune_synthetic_ids.txt
  python -m biomedrag.eval.prune --ids s34784299 s30064232

A copy of the file as it was is saved first to data/synthetic_before_prune.jsonl (not committed).
Running it twice is safe: ids that are already gone are only reported.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

from biomedrag import config
from biomedrag.eval.questions import read_jsonl, write_jsonl

TARGET = config.EVAL / 'synthetic.jsonl'
BACKUP = config.DATA / 'synthetic_before_prune.jsonl'


def parse_ids(text: str) -> list[str]:
    """Ids separated by spaces, commas or new lines. Anything after a # on a line is a comment."""
    ids: list[str] = []
    for line in text.splitlines():
        line = line.split('#', 1)[0]
        ids += [t for t in re.split(r'[\s,]+', line) if t]
    return list(dict.fromkeys(ids))


def prune(ids: list[str], path: Path = TARGET, backup: Path = BACKUP) -> tuple[int, int, list[str]]:
    """Remove the rows whose id is in ids. Returns (kept, removed, ids that were not in the file)."""
    rows = read_jsonl(path)
    present = {r['id'] for r in rows}
    missing = [i for i in ids if i not in present]
    drop = set(ids)
    kept = [r for r in rows if r["id"] not in drop]
    if len(kept) < len(rows):
        if not backup.exists():       # keep the FIRST backup: the untouched original
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, backup)
        write_jsonl(kept, path)
    return len(kept), len(rows) - len(kept), missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', help='text file with ids to delete')
    ap.add_argument('--ids', nargs='*', default=[], help='ids to delete, e.g. s34784299')
    args = ap.parse_args(argv)
    ids = list(args.ids)
    if args.file:
        ids += parse_ids(Path(args.file).read_text(encoding='utf-8'))
    ids = list(dict.fromkeys(ids))
    if not ids:
        print('Give ids with --file or --ids.')
        return 1
    if not TARGET.exists():
        print(f'{TARGET} not found.')
        return 1
    kept, removed, missing = prune(ids, TARGET, BACKUP)
    print(f'Removed {removed}, kept {kept} questions in {TARGET.name}.')
    if removed:
        print(f'Original saved at {BACKUP}')
    if missing:
        print(f'Not found (already removed, or a typo): {", ".join(missing)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
