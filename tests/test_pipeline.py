import pytest
from fakes import FakeLLM, StubRetriever, make_results

from biomedrag.generate.llm import LLMError
from biomedrag.generate.pipeline import REFUSALS, RAGPipeline
from biomedrag.generate.prompt import INSUFFICIENT

GOOD = 'IDH mutation predicts better survival [PMID:1]. MGMT predicts response [PMID:2].'


def make(replies, scores=(7.5, 6.0, 4.5), **kwargs):
    llm = FakeLLM(replies)
    retriever = StubRetriever(make_results(scores))
    return RAGPipeline(retriever, llm, refuse_below=3.0, **kwargs), llm, retriever


def test_a_good_answer_is_returned_with_its_sources_marked():
    pipe, llm, _ = make([GOOD])
    a = pipe.answer('Does IDH matter?')
    assert a.status == 'answered' and a.answered
    assert a.text == GOOD
    assert [(s.pmid, s.cited) for s in a.sources] == [('1', True), ('2', True), ('3', False)]
    assert a.retries == 0 and a.report.ok
    assert len(llm.calls) == 1


def test_the_model_sees_every_retrieved_paper():
    pipe, llm, _ = make([GOOD])
    pipe.answer('Does IDH matter?')
    user = llm.calls[0][1]['content']
    assert 'Does IDH matter?' in user
    assert all(f'[PMID:{p}]' in user for p in ('1', '2', '3'))


def test_a_low_best_score_refuses_without_calling_the_model():
    pipe, llm, _ = make([], scores=(2.5, 2.0, 1.0))
    a = pipe.answer('What is the best diabetes drug?')
    assert a.status == 'refused_low_score' and not a.answered
    assert a.text == REFUSALS['refused_low_score']
    assert llm.calls == []                       # no model call: nothing to pay for
    assert a.top_score == 2.5 and len(a.sources) == 3


def test_a_score_exactly_at_the_threshold_is_allowed_through():
    pipe, llm, _ = make([GOOD], scores=(3.0, 2.0, 1.0))
    assert pipe.answer('q').status == 'answered'
    assert len(llm.calls) == 1


def test_no_retrieved_papers_refuses():
    llm = FakeLLM([])
    a = RAGPipeline(StubRetriever([]), llm).answer('anything')
    assert a.status == 'refused_no_results' and a.top_score is None
    assert llm.calls == []


def test_the_model_can_say_the_evidence_is_insufficient():
    pipe, llm, _ = make([INSUFFICIENT])
    a = pipe.answer('How much does surgery cost in a private hospital?')
    assert a.status == 'refused_by_model'
    assert a.text == REFUSALS['refused_by_model']
    assert len(llm.calls) == 1                   # a refusal is final, not retried
    assert not any(s.cited for s in a.sources)


def test_insufficient_evidence_is_recognised_even_with_extra_words():
    pipe, _, _ = make(['Answer: insufficient_evidence.'])
    assert pipe.answer('q').status == 'refused_by_model'


def test_an_invented_citation_triggers_one_retry_with_feedback():
    pipe, llm, _ = make(['Made up [PMID:999].', GOOD])
    a = pipe.answer('Does IDH matter?')
    assert a.status == 'answered' and a.retries == 1
    assert len(llm.calls) == 2
    retry = llm.calls[1]
    assert retry[-2] == {'role': 'assistant', 'content': 'Made up [PMID:999].'}
    assert '999' in retry[-1]['content']         # the feedback names the bad PMID
    assert 'PMID:1' in retry[-1]['content']      # ...and lists the allowed ones


def test_an_answer_without_any_citation_also_triggers_a_retry():
    pipe, llm, _ = make(['IDH is good for survival.', GOOD])
    a = pipe.answer('q')
    assert a.status == 'answered' and a.retries == 1
    assert 'did not cite' in llm.calls[1][-1]['content']


def test_citations_that_stay_invalid_after_the_retry_lead_to_a_refusal():
    pipe, llm, _ = make(['Bad [PMID:999].', 'Still bad [PMID:888].'])
    a = pipe.answer('q')
    assert a.status == 'refused_bad_citations'
    assert a.text == REFUSALS['refused_bad_citations']
    assert '999' not in a.text and '888' not in a.text      # the bad answer is never shown
    assert len(llm.calls) == 2 and a.retries == 1
    assert a.report.invalid == ['888']


def test_the_model_can_decline_on_the_retry_too():
    pipe, _, _ = make(['Bad [PMID:999].', INSUFFICIENT])
    assert pipe.answer('q').status == 'refused_by_model'


def test_max_retries_zero_never_asks_twice():
    pipe, llm, _ = make(['Bad [PMID:999].'], max_retries=0)
    assert pipe.answer('q').status == 'refused_bad_citations'
    assert len(llm.calls) == 1


def test_a_mixed_valid_and_invalid_answer_is_not_accepted():
    pipe, _, _ = make(['Real [PMID:1]. Fake [PMID:999].', GOOD])
    a = pipe.answer('q')
    assert a.retries == 1 and a.text == GOOD


def test_modes_without_scores_skip_the_threshold():
    llm = FakeLLM([GOOD])
    results = [(c, None) for c, _ in make_results()]
    pipe = RAGPipeline(StubRetriever(results), llm, refuse_below=100.0, mode='hybrid')
    a = pipe.answer('q')
    assert a.status == 'answered' and a.top_score is None


def test_the_pipeline_asks_the_retriever_for_the_configured_mode_and_count():
    pipe, _, retriever = make([GOOD], mode='hybrid_rerank', k_final=2)
    a = pipe.answer('q')
    assert retriever.calls == [{'query': 'q', 'mode': 'hybrid_rerank', 'k_final': 2}]
    assert len(a.sources) == 2


def test_model_errors_are_not_swallowed():
    class Broken:
        name, model = 'broken', 'x'

        def complete(self, messages):
            raise LLMError('cannot reach the model')

    pipe = RAGPipeline(StubRetriever(make_results()), Broken(), refuse_below=3.0)
    with pytest.raises(LLMError, match='cannot reach'):
        pipe.answer('q')


def test_the_answer_can_be_turned_into_plain_data_for_saving():
    pipe, _, _ = make([GOOD])
    d = pipe.answer('q').to_dict()
    assert d['status'] == 'answered' and d['sources'][0]['pmid'] == '1'
    assert d['report']['valid'] == ['1', '2']


def test_the_raw_model_replies_are_kept_for_debugging():
    pipe, _, _ = make(['Bad [PMID:999].', GOOD])
    assert pipe.answer('q').attempts == ['Bad [PMID:999].', GOOD]


def test_the_raw_replies_are_kept_even_when_the_answer_is_refused():
    pipe, _, _ = make(['Bad [PMID:999].', 'Still bad [PMID:888].'])
    a = pipe.answer('q')
    assert a.status == 'refused_bad_citations'
    assert a.attempts == ['Bad [PMID:999].', 'Still bad [PMID:888].']
    assert a.to_dict()['attempts'] == a.attempts
