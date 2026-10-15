"""Question in, cited answer (or an honest refusal) out.

  1. retrieve the best papers (hybrid search + reranker)
  2. refuse if even the best paper scores too low        -> clearly off-topic questions, no model call
  3. ask the language model, which may reply INSUFFICIENT_EVIDENCE  -> on-topic but unanswered questions
  4. check the citations; if bad, ask once more with feedback
  5. still bad -> refuse instead of showing citations we cannot verify
"""
import time
from dataclasses import asdict, dataclass, field

from biomedrag import config
from biomedrag.generate.citations import CitationReport, check_citations
from biomedrag.generate.llm import get_llm
from biomedrag.generate.prompt import INSUFFICIENT, build_messages
from biomedrag.retrieve.retriever import Retriever

REFUSALS = {
    'refused_no_results': 'I found no papers related to this question in the indexed abstracts.',
    'refused_low_score': 'I could not find evidence for this question in the indexed PubMed '
                         'abstracts, so I will not answer it.',
    'refused_by_model': 'The retrieved papers do not contain enough information to answer this '
                        'question.',
    'refused_bad_citations': 'I could not write an answer whose citations I was able to verify, '
                             'so I will not show one.',
}


@dataclass
class Source:
    pmid: str
    title: str
    year: str
    score: float | None
    cited: bool = False         # True if the final answer cites this paper


@dataclass
class Answer:
    question: str
    status: str                 # 'answered' or one of the keys of REFUSALS
    text: str                   # the answer, or the refusal message
    sources: list[Source]
    top_score: float | None
    retries: int = 0
    report: CitationReport | None = None
    retrieve_seconds: float = 0.0
    llm_seconds: float = 0.0
    attempts: list[str] = field(default_factory=list)   # the model's raw replies, for debugging

    @property
    def answered(self) -> bool:
        return self.status == 'answered'

    def to_dict(self) -> dict:
        return asdict(self)


def is_refusal(text: str) -> bool:
    return INSUFFICIENT in text.upper()


def _feedback(report: CitationReport, allowed: list[str]) -> str:
    ids = ', '.join(f'PMID:{p}' for p in allowed)
    if report.invalid:
        bad = ', '.join(report.invalid)
        problem = f'Your answer cited PMIDs that were not in the passages ({bad}).'
    else:
        problem = 'Your answer did not cite any passage.'
    return (f'{problem} Rewrite the answer. Cite each fact like [PMID:12345678], using only these '
            f'PMIDs: {ids}. If the passages cannot answer the question, reply {INSUFFICIENT}.')


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm, refuse_below: float = config.REFUSE_BELOW,
                 mode: str = 'hybrid_rerank', k_final: int = config.K_FINAL,
                 max_retries: int = config.MAX_RETRIES):
        self.retriever, self.llm = retriever, llm
        self.refuse_below, self.mode = refuse_below, mode
        self.k_final, self.max_retries = k_final, max_retries

    @classmethod
    def load(cls, backend: str | None = None, strategy: str = 'abstract', **kwargs) -> 'RAGPipeline':
        return cls(Retriever.load(strategy), get_llm(backend), **kwargs)

    def answer(self, question: str) -> Answer:
        t = time.perf_counter()
        results = self.retriever.retrieve(question, mode=self.mode, k_final=self.k_final)
        retrieve_seconds = time.perf_counter() - t

        sources = [Source(c['pmid'], c['title'], c['year'], s) for c, s in results]
        top = results[0][1] if results else None

        attempts: list[str] = []

        def finish(status: str, text: str | None = None, **extra) -> Answer:
            return Answer(question, status, text or REFUSALS[status], sources, top,
                          retrieve_seconds=retrieve_seconds, attempts=attempts, **extra)

        if not results:
            return finish('refused_no_results')
        if top is not None and top < self.refuse_below:             # strictly below: equal passes
            return finish('refused_low_score')

        chunks = [c for c, _ in results]
        allowed = [c['pmid'] for c in chunks]
        messages = build_messages(question, chunks)

        t = time.perf_counter()
        raw = self.llm.complete(messages)
        attempts.append(raw)
        retries = 0
        while True:
            if is_refusal(raw):
                return finish('refused_by_model', retries=retries, llm_seconds=time.perf_counter() - t)
            report = check_citations(raw, set(allowed))
            if report.ok:
                for s in sources:
                    s.cited = s.pmid in report.valid
                return finish('answered', raw.strip(), retries=retries, report=report,
                              llm_seconds=time.perf_counter() - t)
            if retries >= self.max_retries:
                return finish('refused_bad_citations', retries=retries, report=report,
                              llm_seconds=time.perf_counter() - t)
            messages = messages + [{'role': 'assistant', 'content': raw},
                                   {'role': 'user', 'content': _feedback(report, allowed)}]
            raw = self.llm.complete(messages)
            attempts.append(raw)
            retries += 1
