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
