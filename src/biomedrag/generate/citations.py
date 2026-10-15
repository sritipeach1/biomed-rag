"""Checks the citations in an answer.

The model is told to cite like [PMID:12345678]. A model can still cite a paper it was never given,
or forget to cite. This module finds every cited PMID and compares it with the papers that were
actually retrieved.

What this proves, and what it does not: a citation that points at a retrieved paper is *valid*.
It does not prove the paper really says what the sentence claims. That second question
("faithfulness") is measured separately in Step 5.
"""
import re
from dataclasses import dataclass

# [PMID:1] is what we ask for. Models also write (PMID:1), full-width 【PMID:1】 or a bare "PMID: 1";
# all of them name a paper we can check, so all of them are understood.
BRACKET = re.compile(r'[\[(【]([^\[\]()【】]*)[\])】]')
BARE = re.compile(r'\bPMID\s*[:#]?\s*(\d+)', re.IGNORECASE)
NUMBER = re.compile(r'\d+')
# A sentence end followed by citations ("... fact. [PMID:1]") -> move the citations before the dot.
TRAILING_CITES = re.compile(
    r'([.!?])\s+((?:[\[(【][^\[\]()【】]*pmid[^\[\]()【】]*[\])】]\s*)+)', re.IGNORECASE)
SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z\[(【])')


def extract_citations(text: str) -> list[str]:
    """Every PMID cited in the text, in order of first appearance, without repeats.
    Understands [PMID:1], [PMID: 1], [pmid:1], (PMID:1), 【PMID:1】, [PMID:1, PMID:2], [PMID:1, 2]
    and a bare "PMID: 1"."""
    at: dict[int, str] = {}                             # position in the text -> PMID
    for match in BRACKET.finditer(text):
        inner = match.group(1)
        if 'pmid' not in inner.lower():
            continue                                    # something like [1] is not one of ours
        for number in NUMBER.finditer(inner):
            at[match.start(1) + number.start()] = number.group()
    for match in BARE.finditer(text):
        at[match.start(1)] = match.group(1)
    found: list[str] = []
    for pos in sorted(at):
        if at[pos] not in found:
            found.append(at[pos])
    return found


def split_claims(text: str) -> list[str]:
    """Split an answer into sentences, keeping each sentence's citations attached to it."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = TRAILING_CITES.sub(lambda m: f' {m.group(2).strip()}{m.group(1)} ', text)
    parts = [p.strip() for p in SENTENCE_END.split(text)]
    return [p for p in parts if re.search(r'[A-Za-z]', p)]


@dataclass
class CitationReport:
    cited: list[str]            # every PMID the answer cites
    valid: list[str]            # ...that were among the retrieved papers
    invalid: list[str]          # ...that were NOT (invented or mistyped)
    n_sentences: int
    n_cited_sentences: int      # sentences with at least one citation

    @property
    def ok(self) -> bool:
        """Usable: it cites something, and everything it cites was really retrieved."""
        return bool(self.cited) and not self.invalid

    @property
    def coverage(self) -> float:
        """Share of sentences that carry a citation. Reported, not enforced: models often leave a
        closing summary sentence uncited, and rejecting those would cause pointless retries."""
        return self.n_cited_sentences / self.n_sentences if self.n_sentences else 0.0


def check_citations(text: str, allowed: set[str]) -> CitationReport:
    cited = extract_citations(text)
    sentences = split_claims(text)
    return CitationReport(
        cited=cited,
        valid=[p for p in cited if p in allowed],
        invalid=[p for p in cited if p not in allowed],
        n_sentences=len(sentences),
        n_cited_sentences=sum(1 for s in sentences if extract_citations(s)),
    )
