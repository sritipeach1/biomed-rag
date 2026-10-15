from biomedrag.retrieve.hybrid import collapse_to_pmid, rrf


def test_rrf_puts_items_found_by_both_searches_first():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] in ([1, 3], [3, 1])


def test_rrf_includes_items_from_either_list_once():
    merged = rrf([[1, 2], [3, 2]])
    assert sorted(merged) == [1, 2, 3]
    assert merged[0] == 2                              # the only item in both lists


def test_rrf_respects_top():
    assert len(rrf([[1, 2, 3, 4], [5, 6, 7, 8]], top=3)) == 3


def test_rrf_with_empty_lists():
    assert rrf([[], []]) == []
    assert rrf([[7], []]) == [7]


def test_collapse_keeps_the_first_passage_of_each_paper():
    chunks = [{'pmid': 'a', 'text': 'x'}, {'pmid': 'a', 'text': 'y'}, {'pmid': 'b', 'text': 'z'}]
    assert [c['text'] for c in collapse_to_pmid(chunks)] == ['x', 'z']
