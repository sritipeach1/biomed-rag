import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('push_to_space', ROOT / 'deploy' / 'push_to_space.py')
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def test_assemble_copies_only_what_the_space_needs(tmp_path):
    push.assemble(tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {'Dockerfile', 'pyproject.toml', 'src', 'static', 'README.md'}
    assert (tmp_path / 'README.md').read_text(encoding='utf-8').startswith('---\ntitle:')
    assert (tmp_path / 'src' / 'biomedrag' / 'api' / 'app.py').exists()
    assert not list(tmp_path.rglob('__pycache__'))


def test_without_a_token_it_skips_quietly(monkeypatch, capsys):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.delenv('SPACE_REPO', raising=False)
    assert push.main() == 0 and 'Skipping' in capsys.readouterr().out


def test_the_dockerfile_installs_cpu_torch_before_the_package_and_serves_on_7860():
    text = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    assert text.index('whl/cpu') < text.index('pip install --no-cache-dir -e .')
    assert 'PORT=7860' in text and 'USER user' in text


def test_nothing_secret_or_heavy_goes_into_the_image():
    ignored = (ROOT / '.dockerignore').read_text(encoding='utf-8').split()
    assert {'.env', 'data', 'index', '.git'} <= set(ignored)
