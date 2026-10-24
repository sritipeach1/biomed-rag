"""Get the pipeline ready when the server starts.

On your own computer the index is in index/abstract. On a fresh container it is not there, so it is
downloaded once from a private Hugging Face dataset (INDEX_REPO) using a read token (HF_TOKEN).
"""
import os
from pathlib import Path

from biomedrag import config
from biomedrag.generate.pipeline import RAGPipeline

NEEDED = ('chunks.jsonl', 'bm25.pkl', 'dense.faiss')


def ensure_index(name: str = 'abstract', index_root: Path = config.INDEX, repo: str | None = None,
                 token: str | None = None, download=None) -> Path:
    """Return the index folder, downloading it first if the files are missing.
    `download` is only passed in by tests."""
    folder = index_root / name
    if all((folder / f).exists() for f in NEEDED):
        return folder
    repo = repo or config.INDEX_REPO
    if not repo:
        raise RuntimeError(f'The index is missing in {folder} and INDEX_REPO is not set, so it cannot '
                           f'be downloaded. Build it with: python -m biomedrag.index.build --strategy {name}')
    if download is None:
        from huggingface_hub import snapshot_download as download
    download(repo_id=repo, repo_type='dataset', local_dir=str(folder),
             token=token or os.environ.get('HF_TOKEN') or None)
    missing = [f for f in NEEDED if not (folder / f).exists()]
    if missing:
        raise RuntimeError(f'Downloaded {repo} but these files are still missing: {", ".join(missing)}')
    return folder


def load_pipeline(strategy: str = 'abstract') -> RAGPipeline:
    ensure_index(strategy)
    return RAGPipeline.load(config.LLM_BACKEND, strategy, refuse_below=config.REFUSE_BELOW,
                            prompt_version=config.PROMPT_VERSION)
