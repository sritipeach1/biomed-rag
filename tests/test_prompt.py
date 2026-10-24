from fakes import make_chunks

from biomedrag.generate.prompt import INSUFFICIENT, SYSTEM_PROMPT, build_messages


def test_messages_have_a_system_and_a_user_part():
    msgs = build_messages('What is IDH?', make_chunks()[:2])
    assert [m['role'] for m in msgs] == ['system', 'user']


def test_every_passage_appears_with_its_pmid_tag():
    chunks = make_chunks()[:5]
    user = build_messages('q', chunks)[1]['content']
    for c in chunks:
        assert f'[PMID:{c["pmid"]}]' in user
        assert c['text'] in user


def test_the_question_is_in_the_user_message():
    assert 'Does MGMT matter?' in build_messages('Does MGMT matter?', make_chunks()[:1])[1]['content']


def test_the_system_prompt_teaches_the_refusal_word_and_the_citation_format():
    assert INSUFFICIENT in SYSTEM_PROMPT
    assert '[PMID:12345678]' in SYSTEM_PROMPT
    assert 'ONLY' in SYSTEM_PROMPT


def test_prompt_v2_is_stricter_but_keeps_the_refusal_word_and_citation_format():
    from biomedrag.generate.prompt import PROMPTS, SYSTEM_PROMPT_V2
    assert PROMPTS['v1'] == SYSTEM_PROMPT and PROMPTS['v2'] == SYSTEM_PROMPT_V2 != SYSTEM_PROMPT
    assert INSUFFICIENT in SYSTEM_PROMPT_V2 and '[PMID:12345678]' in SYSTEM_PROMPT_V2
    assert 'not a question about brain tumour' in SYSTEM_PROMPT_V2


def test_the_version_chooses_the_system_prompt_and_unknown_versions_fail_loudly():
    import pytest

    from biomedrag.generate.prompt import SYSTEM_PROMPT_V2
    assert build_messages('q', make_chunks()[:1])[0]['content'] == SYSTEM_PROMPT
    assert build_messages('q', make_chunks()[:1], 'v2')[0]['content'] == SYSTEM_PROMPT_V2
    with pytest.raises(ValueError):
        build_messages('q', make_chunks()[:1], 'v9')


def test_prompt_v3_demands_a_citation_on_every_sentence_and_bans_unsupported_superlatives():
    from biomedrag.generate.prompt import PROMPTS, SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3
    assert PROMPTS['v3'] == SYSTEM_PROMPT_V3 not in (SYSTEM_PROMPT, SYSTEM_PROMPT_V2)
    assert 'EVERY sentence must end with at least one citation' in SYSTEM_PROMPT_V3
    assert 'closing summary' in SYSTEM_PROMPT_V3 and '"most effective"' in SYSTEM_PROMPT_V3
    assert INSUFFICIENT in SYSTEM_PROMPT_V3 and '[PMID:12345678]' in SYSTEM_PROMPT_V3
    assert 'not a question about brain tumour' in SYSTEM_PROMPT_V3          # keeps the v2 off-topic rule
    assert build_messages('q', make_chunks()[:1], 'v3')[0]['content'] == SYSTEM_PROMPT_V3
