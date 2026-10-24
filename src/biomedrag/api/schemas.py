"""What the API accepts and returns, and how a pipeline Answer becomes a response."""
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from biomedrag.generate.pipeline import Answer

PUBMED = 'https://pubmed.ncbi.nlm.nih.gov/{}/'
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=500)]


class AskRequest(BaseModel):
    question: Question


class SourceOut(BaseModel):
    pmid: str
    title: str
    year: str
    cited: bool
    url: str


class AskResponse(BaseModel):
    question: str
    status: str                 # 'answered', or the reason the question was declined
    answered: bool
    text: str
    sources: list[SourceOut]
    seconds: float
    from_cache: bool = False


def normalise_citations(text: str) -> str:
    """Some models write full-width brackets (【PMID:1】). Show every citation as [PMID:1]."""
    return text.replace('【', '[').replace('】', ']')


def present(answer: Answer) -> AskResponse:
    """The public view of an answer. Paper titles and PubMed links are shown; the abstracts
    themselves are not sent, because PubMed is the right place to read them."""
    sources = []
    if answer.answered:
        sources = [SourceOut(pmid=s.pmid, title=s.title, year=s.year, cited=s.cited,
                             url=PUBMED.format(s.pmid)) for s in answer.sources]
    return AskResponse(question=answer.question, status=answer.status, answered=answer.answered,
                       text=normalise_citations(answer.text), sources=sources,
                       seconds=round(answer.retrieve_seconds + answer.llm_seconds, 1))
