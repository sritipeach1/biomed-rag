"""Copy the files the demo needs into a Hugging Face Space (SDK: Docker) and upload them.

Run by the GitHub Action after the tests pass. Needs these environment variables:
    HF_TOKEN    a Hugging Face token with WRITE access
    SPACE_REPO  the Space, like  your-hf-name/biomed-rag
If either is missing, it prints why and exits without error, so the tests still count.
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


def assemble(dest: Path, root: Path = ROOT) -> None:
    """Put exactly what the Space needs into `dest`. The Space's README is the one with its front matter."""
    for name in FILES:
        shutil.copy2(root / name, dest / name)
    for name in FOLDERS:
        shutil.copytree(root / name, dest / name, ignore=SKIP)
    shutil.copy2(root / 'deploy' / 'SPACE_README.md', dest / 'README.md')


def main() -> int:
    token, repo = os.environ.get('HF_TOKEN', '').strip(), os.environ.get('SPACE_REPO', '').strip()
    if not token or not repo:
        print('Skipping the Space upload: HF_TOKEN (secret) and HF_SPACE (variable) are not both set.')
        return 0
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type='space', space_sdk='docker', exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        assemble(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo, repo_type='space',
                          commit_message='Update from GitHub')
    print(f'Uploaded to https://huggingface.co/spaces/{repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
