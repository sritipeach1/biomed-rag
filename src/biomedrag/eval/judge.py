r"""Ask a second model: is each cited sentence really supported by the paper it cites?

  python -m biomedrag.eval.judge --split dev --run groq-openai-gpt-oss-120b-v2
  python -m biomedrag.eval.judge --split dev --run groq-openai-gpt-oss-120b-v2 --report-only

The citation checker only proves that a cited PMID was retrieved. This checks the harder thing: whether
the SENTENCE is backed by that paper. Every cited sentence gets one verdict:
  supported    the cited paper states the claim, or clearly implies it
  partial      part of the claim is backed, but a detail, number or conclusion is not
  unsupported  the paper does not back it, contradicts it, or is about something else
Sentences without any citation are counted separately ("uncited"); they cannot be checked.

Use a DIFFERENT model from the one that wrote the answers (default: openai/gpt-oss-20b), and check its
verdicts against your own with  python -m biomedrag.eval.labels  before you trust the numbers.
Verdicts are saved as they come (data/answers/judge_*.jsonl), so a stopped run just continues.
"""
import argparse
import json
import random
import re
import sys
import time

from biomedrag import config
from biomedrag.eval import metrics
from biomedrag.eval.answers import ANSWERS, DAILY_LIMIT, answers_path, load_done
from biomedrag.generate.citations import extract_citations, split_claims
from biomedrag.generate.llm import LLMError, get_llm

VERDICTS = ('supported', 'partial', 'unsupported')
DEFAULT_JUDGE = 'openai/gpt-oss-20b'

SYSTEM = """You are a careful fact-checker for biomedical literature answers. You are given numbered sentences \
from an answer, each with the PMIDs it cites, and the text of the cited PubMed passages.

For each sentence, using ONLY the passages that sentence cites (ignore your own knowledge), choose one verdict:
- "supported": a cited passage states the claim, or clearly implies it.
- "partial": part of the claim is backed, but some detail, number, comparison or conclusion is not in the cited passages.
- "unsupported": the cited passages do not back the claim, contradict it, or are about something else.

Reply with JSON only, no other text, in exactly this form:
{"verdicts": [{"n": 1, "verdict": "supported", "reason": "one short phrase"}]}
Include every numbered sentence."""


def judge_tag(model: str) -> str:
    return re.sub(r'[^A-Za-z0-9.]+', '-', model).strip('-')


def judge_path(split: str, run: str, model: str):
    return ANSWERS / f'judge_{split}_{run}__{judge_tag(model)}.jsonl'


def claims_of(row: dict) -> list[dict]:
    """The answer's sentences, each with the PMIDs it cites (empty list = uncited)."""
    return [{'n': i, 'text': s, 'pmids': extract_citations(s)}
            for i, s in enumerate(split_claims(row['text']), start=1)]


def build_messages(claims: list[dict], passages: dict[str, str]) -> list[dict]:
    cited = [c for c in claims if c['pmids']]
    wanted = list(dict.fromkeys(p for c in cited for p in c['pmids']))
    passage_block = '\n\n'.join(f'[PMID:{p}]\n{passages[p]}' for p in wanted if p in passages)
    sentence_block = '\n'.join(f'{c["n"]}. {c["text"]}   (cites: {", ".join(c["pmids"])})' for c in cited)
    user = f'Passages:\n\n{passage_block}\n\nSentences to check:\n{sentence_block}'
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def parse_verdicts(reply: str) -> dict[int, dict]:
    """{sentence number: {'verdict', 'reason'}} from the model's reply. Tolerates code fences, thinking tags
    and extra words around the JSON. Anything it cannot read is simply left out."""
    text = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    for item in data.get('verdicts', []) if isinstance(data, dict) else []:
        try:
            n, verdict = int(item['n']), str(item['verdict']).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if verdict in VERDICTS:
            out[n] = {'verdict': verdict, 'reason': str(item.get('reason', ''))[:200]}
    return out


def judge_answer(llm, row: dict) -> dict:
    """Judge one answered row. Returns the record that is saved."""
    claims = claims_of(row)
    passages = {s['pmid']: s['text'] for s in row['sources']}
    to_check = [c for c in claims if c['pmids']]
    verdicts: dict[int, dict] = {}
    raw = ''
    if to_check:
        raw = llm.complete(build_messages(claims, passages))
        verdicts = parse_verdicts(raw)
    sentences = []
    for c in claims:
        v = verdicts.get(c['n'])
        label = 'uncited' if not c['pmids'] else (v['verdict'] if v else 'unparsed')
        sentences.append({**c, 'verdict': label, 'reason': v['reason'] if v else ''})
    return {'id': row['id'], 'question': row['question'], 'sentences': sentences, 'raw': raw}


def judge_run(llm, rows: list[dict], path, sleep: float = 0.0, max_errors: int = 3, log=print) -> tuple[int, int]:
    """Judge every answered row not in `path` yet. Returns (new, failed)."""
    done = load_done(path)
    todo = [r for r in rows if r['status'] == 'answered' and r['id'] not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = failed = in_a_row = 0
    with open(path, 'a', encoding='utf-8') as out:
        for i, row in enumerate(todo, start=1):
            try:
                rec = judge_answer(llm, row)
            except LLMError as e:
                failed += 1
                in_a_row += 1
                log(f'  [{i}/{len(todo)}] {row["id"]}: model error, will be retried next run: {e}')
                if any(w in str(e) for w in DAILY_LIMIT):
                    log('  the daily token allowance is used up. Run the same command again later.')
                    break
                if in_a_row >= max_errors:
                    log(f'  stopping after {max_errors} errors in a row.')
                    break
                continue
            in_a_row = 0
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            new += 1
            counts = {v: sum(1 for s in rec['sentences'] if s['verdict'] == v) for v in (*VERDICTS, 'uncited', 'unparsed')}
            log(f'  [{i}/{len(todo)}] {row["id"]}: ' + ', '.join(f'{k} {n}' for k, n in counts.items() if n))
            if sleep:
                time.sleep(sleep)
    return new, failed


def controls_path(split: str, run: str, model: str):
    return ANSWERS / f'judgecontrols_{split}_{run}__{judge_tag(model)}.jsonl'


def make_controls(rows: list[dict], n: int = 10, seed: int = 7) -> list[dict]:
    """Answer-shaped rows with ONE sentence whose citation was swapped for the retrieved paper it scored lowest
    against (a paper it did not cite). A good judge should NOT call these supported. It is a check that the
    judge can say no at all: a judge that approves 98% of true sentences might simply approve everything."""
    rng = random.Random(seed)
    pool = []
    for r in rows:
        if r['status'] != 'answered':
            continue
        for c in claims_of(r):
            others = [s for s in r['sources'] if s['pmid'] not in c['pmids']]
            if len(c['pmids']) == 1 and others:
                pool.append((r, c, others))
    rng.shuffle(pool)
    out, used = [], set()
    for r, c, others in pool:
        if r['id'] in used:
            continue
        used.add(r['id'])
        wrong = min(others, key=lambda s: s['score'] if s['score'] is not None else 0.0)
        out.append({'id': f'c_{r["id"]}', 'question': r['question'], 'status': 'answered', 'sources': r['sources'],
                    'text': c['text'].replace(c['pmids'][0], wrong['pmid']), 'control_of': r['id']})
        if len(out) == n:
            break
    return out


def control_report(records: list[dict]) -> list[str]:
    checked = [x for r in records for x in r['sentences'] if x['verdict'] in VERDICTS]
    if not checked:
        return ['no control sentences judged yet']
    caught = [x for x in checked if x['verdict'] != 'supported']
    lines = [(f'wrong-paper control sentences: the judge flagged {len(caught)} of {len(checked)} '
              f'({len(caught) / len(checked):.0%}) as not fully supported')]
    lines += [f'  missed: {x["text"][:140]}' for x in checked if x['verdict'] == 'supported'][:5]
    lines.append('(a few can be honest misses: the swapped paper may happen to say something similar)')
    return lines


def summarise(records: list[dict]) -> dict:
    """Faithfulness numbers from saved verdicts."""
    sentences = [s for r in records for s in r['sentences']]
    checked = [s for s in sentences if s['verdict'] in VERDICTS]
    per_answer = []
    for r in records:
        c = [s for s in r['sentences'] if s['verdict'] in VERDICTS]
        if c:
            per_answer.append(sum(1 for s in c if s['verdict'] == 'supported') / len(c))
    return {
        'answers': len(records), 'sentences': len(sentences), 'checked': len(checked),
        'uncited': [{**x, 'id': r['id']} for r in records for x in r['sentences'] if x['verdict'] == 'uncited'],
        'unparsed': sum(1 for s in sentences if s['verdict'] == 'unparsed'),
        'rate': {v: metrics.mean([1.0 if s['verdict'] == v else 0.0 for s in checked]) for v in VERDICTS},
        'per_answer_supported': per_answer,
        'fully_supported_answers': [1.0 if x == 1.0 else 0.0 for x in per_answer],
    }


def report(records: list[dict], title: str) -> list[str]:
    s = summarise(records)
    if not s['checked']:
        return [f'**{title}**', '', 'no checked sentences yet']
    lo, hi = metrics.bootstrap_ci(s['per_answer_supported'])
    flo, fhi = metrics.bootstrap_ci(s['fully_supported_answers'])
    lines = [f'**{title}**', '',
             (f'answers judged: {s["answers"]}   sentences: {s["sentences"]}   checked (cited): {s["checked"]}   '
              f'unreadable verdicts: {s["unparsed"]}'), '',
             '| measure | value |', '|---|---|',
             f'| cited sentences supported | {s["rate"]["supported"]:.0%} |',
             f'| cited sentences partial | {s["rate"]["partial"]:.0%} |',
             f'| cited sentences unsupported | {s["rate"]["unsupported"]:.0%} |',
             f'| answers: mean share of cited sentences supported | {metrics.mean(s["per_answer_supported"]):.0%} [{lo:.0%}, {hi:.0%}] (n={len(s["per_answer_supported"])}) |',
             f'| answers where EVERY cited sentence is supported | {metrics.mean(s["fully_supported_answers"]):.0%} [{flo:.0%}, {fhi:.0%}] |',
             f'| sentences with no citation at all | {len(s["uncited"])} of {s["sentences"]} |']
    flagged = [(r['id'], x) for r in records for x in r['sentences'] if x['verdict'] in ('partial', 'unsupported')]
    if flagged:
        lines += ['', 'sentences the judge did not fully support (check a few yourself):']
        for qid, x in flagged[:8]:
            lines.append(f'  {qid} [{x["verdict"]}] {x["text"][:160]}  -- {x["reason"]}')
    if s['uncited']:
        lines += ['', 'examples of sentences with no citation:']
        for x in s['uncited'][:5]:
            lines.append(f'  {x["id"]}: {x["text"][:160]}')
    return lines


def save_summary(records: list[dict], split: str, run: str, model: str) -> None:
    """One row per run and judge in eval/results/faithfulness_<split>.csv (replaces that run's old row)."""
    import csv
    s = summarise(records)
    if not s['checked']:
        return
    path = config.RESULTS / f'faithfulness_{split}.csv'
    fields = ['run', 'judge', 'answers', 'checked_sentences', 'supported', 'partial', 'unsupported',
              'uncited_sentences', 'all_sentences', 'answers_fully_supported']
    keep = []
    if path.exists():
        with open(path, newline='', encoding='utf-8') as f:
            keep = [r for r in csv.DictReader(f) if not (r['run'] == run and r['judge'] == model)]
    keep.append({'run': run, 'judge': model, 'answers': s['answers'], 'checked_sentences': s['checked'],
                 'supported': f'{s["rate"]["supported"]:.4f}', 'partial': f'{s["rate"]["partial"]:.4f}',
                 'unsupported': f'{s["rate"]["unsupported"]:.4f}', 'uncited_sentences': len(s['uncited']),
                 'all_sentences': s['sentences'],
                 'answers_fully_supported': f'{metrics.mean(s["fully_supported_answers"]):.4f}'})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    print(f'\nsaved {path}')


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='Check whether cited sentences are supported by their papers.')
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--final', action='store_true', help='required for --split test')
    ap.add_argument('--run', required=True, help='run name printed by answers.py')
    ap.add_argument('--backend', choices=['ollama', 'groq'], default='groq')
    ap.add_argument('--judge-model', default=DEFAULT_JUDGE)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sleep', type=float, default=1.0)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--controls', type=int, default=0, metavar='N',
                    help='instead of judging the answers, judge N sentences whose citation was swapped for a '
                         'wrong paper: the judge should flag them (a check that it can say no)')
    args = ap.parse_args(argv)
    if args.split == 'test' and not args.final:
        sys.exit('The test split is for reporting only; use --final once.')
    src = answers_path(args.split, args.run)
    if not src.exists():
        sys.exit(f'No saved answers at {src}.')
    rows = list(load_done(src).values())
    if args.limit:
        rows = rows[:args.limit]
    if args.controls:
        cpath = controls_path(args.split, args.run, args.judge_model)
        try:
            llm = get_llm(args.backend, model=args.judge_model)
        except (ValueError, LLMError) as e:
            sys.exit(f'Could not start the judge: {e}')
        new, failed = judge_run(llm, make_controls(rows, args.controls), cpath, sleep=args.sleep)
        print(f'\ndone: {new} judged, {failed} failed.' + ('  Run the same command again to retry.' if failed else ''))
        print('\n' + '\n'.join(control_report(list(load_done(cpath).values()) if cpath.exists() else [])))
        return 0
    path = judge_path(args.split, args.run, args.judge_model)
    if not args.report_only:
        if judge_tag(args.judge_model) in args.run:
            print('warning: the judge model is the same one that wrote the answers; prefer a different model.')
        try:
            llm = get_llm(args.backend, model=args.judge_model)
        except (ValueError, LLMError) as e:
            sys.exit(f'Could not start the judge: {e}')
        print(f'judge: {llm.name} / {llm.model}   answers file: {src.name}')
        new, failed = judge_run(llm, rows, path, sleep=args.sleep)
        print(f'\ndone: {new} judged, {failed} failed.' + ('  Run the same command again to retry.' if failed else ''))
    records = list(load_done(path).values()) if path.exists() else []
    print('\n' + '\n'.join(report(records, f'{args.run} judged by {args.judge_model}')))
    save_summary(records, args.split, args.run, args.judge_model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
