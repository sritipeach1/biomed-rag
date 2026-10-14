from Bio.Entrez.Parser import StringElement

from biomedrag.ingest.pubmed import parse


def text(value, **attrs):
    """Build a Biopython string element, as Entrez.read returns for abstract parts."""
    return StringElement(value, 'AbstractText', attrs, 'AbstractText')


def record(abstract_parts, title='A glioma study', year='2024'):
    article = {
        'ArticleTitle': title,
        'Journal': {'Title': 'Neuro-Oncology', 'JournalIssue': {'PubDate': {'Year': year}}},
    }
    if abstract_parts is not None:
        article['Abstract'] = {'AbstractText': abstract_parts}
    return {'MedlineCitation': {'PMID': '12345678', 'Article': article}}


def test_plain_abstract_is_parsed():
    row = parse(record([text('Temozolomide improves survival.')]))
    assert row == {
        'pmid': '12345678',
        'title': 'A glioma study',
        'abstract': 'Temozolomide improves survival.',
        'journal': 'Neuro-Oncology',
        'year': '2024',
    }


def test_structured_abstract_keeps_section_labels():
    row = parse(record([text('We tested X.', Label='BACKGROUND'),
                        text('X helped.', Label='RESULTS')]))
    assert row['abstract'] == 'BACKGROUND: We tested X. RESULTS: X helped.'


def test_missing_abstract_returns_none():
    assert parse(record(None)) is None


def test_missing_year_gives_empty_string():
    rec = record([text('Some text.')])
    rec['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'] = {}
    assert parse(rec)['year'] == ''
