"""The instructions and evidence we hand to the language model."""

INSUFFICIENT = 'INSUFFICIENT_EVIDENCE'

SYSTEM_PROMPT = f"""You answer questions about brain tumour (glioma) research. You may use ONLY the \
numbered PubMed passages given in the user message.

Rules:
1. State only what the passages say. Do not add facts from your own knowledge.
2. After every sentence that states a fact, cite the passage it came from, written exactly like \
[PMID:12345678], using the PMID shown for that passage. Never cite a PMID that is not in the list.
3. If the passages do not contain enough information to answer the question, reply with exactly \
{INSUFFICIENT} and nothing else. Do not guess. Do not answer a different question than the one asked.
4. If the passages disagree, say so and cite both sides.
5. Be concise: 3 to 6 sentences, written for a researcher. This is not medical advice."""


def format_passages(chunks: list[dict]) -> str:
    """One block per paper: its PMID tag, its year, then the text (which starts with the title)."""
    return '\n\n'.join(f'[PMID:{c["pmid"]}] ({c["year"]})\n{c["text"]}' for c in chunks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    user = f'Question: {question}\n\nPassages:\n\n{format_passages(chunks)}'
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
