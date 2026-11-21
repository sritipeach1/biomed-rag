"""Copy the files the demo needs into a Hugging Face Space and upload them.

Run by the GitHub Action after the tests pass. Needs these environment variables:
    HF_TOKEN    a Hugging Face token with WRITE access
    SPACE_REPO  the Space, like  your-hf-name/biomed-rag
    SPACE_KIND  'static' (default, free: the page shows the saved example answers) or
                'docker' (the live service; Hugging Face charges for Docker Spaces)
If HF_TOKEN or SPACE_REPO is missing, it prints why and exits without error, so the tests still count.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['Dockerfile', 'pyproject.toml']
FOLDERS = ['src', 'static']
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.ruff_cache')
KINDS = ('static', 'docker')
LIVE_TAG = '<meta name="biomedrag-mode" content="live">'
STATIC_TAG = '<meta name="biomedrag-mode" content="static">'


def assemble(dest: Path, root: Path = ROOT, kind: str = 'static') -> None:
    """Put exactly what the Space needs into `dest`. The Space's README is the one with its front matter."""
    if kind == 'docker':
        for name in FILES:
            shutil.copy2(root / name, dest / name)
        for name in FOLDERS:
            shutil.copytree(root / name, dest / name, ignore=SKIP)
        shutil.copy2(root / 'deploy' / 'SPACE_README.md', dest / 'README.md')
    elif kind == 'static':
        page = (root / 'static' / 'index.html').read_text(encoding='utf-8')
        if LIVE_TAG not in page:
            raise SystemExit('static/index.html has no mode tag to switch to static mode.')
        cache = root / 'static' / 'demo_cache.json'
        if not cache.exists():
            raise SystemExit('static/demo_cache.json is missing: the static page has nothing to show. '
                             'Run: python -m biomedrag.api.demo_cache --backend groq')
        (dest / 'index.html').write_text(page.replace(LIVE_TAG, STATIC_TAG), encoding='utf-8')
        shutil.copy2(cache, dest / 'demo_cache.json')
        shutil.copy2(root / 'deploy' / 'SPACE_README_STATIC.md', dest / 'README.md')
    else:
        raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')


def main() -> int:
    token, repo = os.environ.get('HF_TOKEN', '').strip(), os.environ.get('SPACE_REPO', '').strip()
    kind = os.environ.get('SPACE_KIND', '').strip() or 'static'
    if not token or not repo:
        print('Skipping the Space upload: HF_TOKEN (secret) and HF_SPACE (variable) are not both set.')
        return 0
    if kind not in KINDS:
        print(f'SPACE_KIND must be one of {KINDS}, not {kind!r}.')
        return 1
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type='space', space_sdk=kind, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        assemble(Path(tmp), kind=kind)
        api.upload_folder(folder_path=tmp, repo_id=repo, repo_type='space',
                          commit_message='Update from GitHub')
    print(f'Uploaded ({kind}) to https://huggingface.co/spaces/{repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
