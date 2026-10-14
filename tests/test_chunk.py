import json
from collections import Counter

import pytest

from biomedrag import config
from biomedrag.ingest.chunk import (
    chunk_abstract,
    chunk_sentences,
    clean,
    load_clean_docs,
    sample_docs,
    split_sentences,
)


def make_doc(n_sentences: int, pmid: str = '1', title: str = 'T') -> dict:
    sentences = [f'Sentence number {k} is about glioma.' for k in range(n_sentences)]
    return {'pmid': pmid, 'title': title, 'year': '2024', 'abstract': ' '.join(sentences)}


# ---------- clean ----------

def test_clean_removes_formatting_tags():
    assert clean('T<sub>1</sub> and <i>in vitro</i> and Ca<sup>2+</sup>') == 'T1 and in vitro and Ca2+'


def test_clean_keeps_comparison_signs():
    text = 'Survival improved (p < 0.05) and risk fell when x > 3.'
    assert clean(text) == text


def test_clean_turns_block_tags_into_spaces():
    assert clean('First part.<br>Second part.') == 'First part. Second part.'


def test_clean_unescapes_entities_and_collapses_whitespace():
    assert clean('AT&amp;T   study\n\nresults') == 'AT&T study results'


# ---------- chunking ----------

def test_abstract_strategy_gives_one_chunk_with_title_in_front():
    rows = chunk_abstract(make_doc(4, title='A title'))
    assert len(rows) == 1
    assert rows[0]['text'].startswith('A title. Sentence number 0')
    assert rows[0]['chunk_id'] == '1_0'


def test_title_ending_in_punctuation_is_not_doubled():
    assert chunk_abstract(make_doc(3, title='What is new?'))[0]['text'].startswith('What is new? Sentence')
    assert chunk_abstract(make_doc(3, title='A study.'))[0]['text'].startswith('A study. Sentence')


@pytest.mark.parametrize('n', range(1, 13))
def test_sentence_windows_never_lose_a_sentence(n):
    rows = chunk_sentences(make_doc(n))
    joined = ' '.join(r['text'] for r in rows)
    for k in range(n):
        assert f'Sentence number {k} is about glioma.' in joined


def test_sentence_windows_have_at_most_three_sentences_and_unique_ids():
    rows = chunk_sentences(make_doc(10))
    assert all(r['text'].count('Sentence number') <= 3 for r in rows)
    ids = [r['chunk_id'] for r in rows]
    assert len(ids) == len(set(ids))


def test_every_sentence_chunk_starts_with_the_title():
    assert all(r['text'].startswith('T. ') for r in chunk_sentences(make_doc(7)))


def test_abbreviations_do_not_end_sentences():
    text = ('Median survival was ca. 15 months in approx. 40% of cases. '
            'No. of patients rose vs. the earlier cohort, i.e. by 12.5%. '
            'Dr. Smith et al. agreed (p < 0.05).')
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == 'Median survival was ca. 15 months in approx. 40% of cases.'
    assert sents[1].startswith('No. of patients')


def test_real_sentence_end_before_capital_letter_is_still_split():
    sents = split_sentences('The answer was no. The next study began.')
    assert len(sents) == 2


def test_one_sentence_abstract_does_not_crash():
    assert len(chunk_sentences(make_doc(1))) == 1


# ---------- cleaning filters and sampling ----------

def write_corpus(path, docs):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(d) + '\n' for d in docs)


def test_filters_remove_short_long_and_duplicate_abstracts(tmp_path):
    ok = make_doc(12, pmid='1')                              # about 70 words
    short = {**make_doc(2, pmid='2')}                        # about 12 words
    too_long = {**make_doc(2, pmid='3'), 'abstract': 'word ' * (config.MAX_WORDS + 1)}
    duplicate = {**ok, 'pmid': '4'}                          # same text, different PMID
    path = tmp_path / 'abstracts.jsonl'
    write_corpus(path, [ok, short, too_long, duplicate])

    stats = Counter()
    kept = list(load_clean_docs(path, stats))

    assert [d['pmid'] for d in kept] == ['1']
    assert stats['read'] == 4
    assert stats['too_short'] == 1
    assert stats['too_long'] == 1
    assert stats['duplicate_text'] == 1
    assert stats['kept'] == 1


def test_sample_is_repeatable_and_has_the_right_size():
    docs = [make_doc(3, pmid=str(i)) for i in range(100)]
    a = sample_docs(docs, n=10, seed=7)
    b = sample_docs(docs, n=10, seed=7)
    c = sample_docs(docs, n=10, seed=8)
    assert [d['pmid'] for d in a] == [d['pmid'] for d in b]
    assert [d['pmid'] for d in a] != [d['pmid'] for d in c]
    assert len(a) == 10


def test_sample_larger_than_corpus_returns_everything():
    docs = [make_doc(3, pmid=str(i)) for i in range(5)]
    assert len(sample_docs(docs, n=100)) == 5
