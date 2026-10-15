import pytest

from biomedrag.generate.citations import check_citations, extract_citations, split_claims


@pytest.mark.parametrize('text, expected', [
    ('IDH helps [PMID:111].', ['111']),
    ('One [PMID:111] and two [PMID:222].', ['111', '222']),
    ('Both [PMID:111, PMID:222].', ['111', '222']),
    ('Spaces [PMID: 111] and [pmid:222].', ['111', '222']),
    ('Short form [PMID:111, 222].', ['111', '222']),
    ('Repeated [PMID:111] and again [PMID:111].', ['111']),
    ('Round brackets (PMID:111) and (PMID: 222, PMID:333).', ['111', '222', '333']),
    ('Plain parentheses (see Fig. 3) are not citations.', []),
    ('Full-width brackets 【PMID:111】 and 【PMID: 222, 333】.', ['111', '222', '333']),
    ('Bare form PMID: 111 and PMID 222 and pmid#333.', ['111', '222', '333']),
    ('Bracket plus bare, same paper: [PMID:111] and PMID: 111.', ['111']),
    ('A word like PMIDs without a number mentions nothing.', []),
    ('No citations here.', []),
    ('Numbered style [1] and [2] is not ours.', []),
    ('A year in brackets [2021] is not a citation.', []),
])
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def test_citations_come_back_in_order_of_first_appearance():
    assert extract_citations('A [PMID:3]. B [PMID:1]. C [PMID:3] [PMID:2].') == ['3', '1', '2']


def test_valid_and_invalid_citations_are_separated():
    r = check_citations('Real [PMID:1]. Invented [PMID:999].', {'1', '2'})
    assert r.cited == ['1', '999']
    assert r.valid == ['1']
    assert r.invalid == ['999']
    assert not r.ok


def test_all_valid_citations_are_ok():
    r = check_citations('Fact [PMID:1]. Another [PMID:2].', {'1', '2', '3'})
    assert r.ok and r.invalid == []


def test_an_answer_without_citations_is_not_ok():
    r = check_citations('IDH mutation improves survival.', {'1'})
    assert not r.ok and r.cited == []


def test_citation_after_the_full_stop_still_counts_for_its_sentence():
    r = check_citations('IDH mutation improves survival. [PMID:1] MGMT predicts response. [PMID:2]',
                        {'1', '2'})
    assert (r.n_sentences, r.n_cited_sentences) == (2, 2)


def test_coverage_counts_uncited_sentences():
    r = check_citations('Cited fact [PMID:1]. Cited again [PMID:1]. A closing remark.', {'1'})
    assert (r.n_sentences, r.n_cited_sentences) == (3, 2)
    assert r.coverage == pytest.approx(2 / 3)


def test_coverage_of_empty_text_is_zero():
    assert check_citations('', {'1'}).coverage == 0.0


def test_sentence_splitting_keeps_figure_references_together():
    parts = split_claims('Survival improved (Fig. 3 shows this) [PMID:1]. Another fact [PMID:2].')
    assert len(parts) == 2


def test_round_bracket_citation_after_the_full_stop_stays_with_its_sentence():
    r = check_citations('IDH improves survival. (PMID:1) MGMT predicts response. (PMID:2)', {'1', '2'})
    assert (r.n_sentences, r.n_cited_sentences) == (2, 2)


def test_full_width_citation_after_the_full_stop_stays_with_its_sentence():
    r = check_citations('IDH improves survival. 【PMID:1】 MGMT predicts response. 【PMID:2】', {'1', '2'})
    assert (r.n_sentences, r.n_cited_sentences) == (2, 2)
