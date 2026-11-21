import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('push_to_space', ROOT / 'deploy' / 'push_to_space.py')
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def fake_root(tmp_path):
    """The real page and Space README, with a small made-up saved-answers file."""
    root = tmp_path / 'repo'
    (root / 'static').mkdir(parents=True)
    (root / 'deploy').mkdir()
    shutil.copy2(ROOT / 'static' / 'index.html', root / 'static' / 'index.html')
    shutil.copy2(ROOT / 'deploy' / 'SPACE_README_STATIC.md', root / 'deploy' / 'SPACE_README_STATIC.md')
    (root / 'static' / 'demo_cache.json').write_text(
        json.dumps([{'question': 'Example question here?', 'response': {'answered': True}}]), encoding='utf-8')
    return root


def test_static_space_gets_only_the_page_the_answers_and_a_readme(tmp_path):
    dest = tmp_path / 'out'
    dest.mkdir()
    push.assemble(dest, root=fake_root(tmp_path), kind='static')
    assert {p.name for p in dest.iterdir()} == {'index.html', 'demo_cache.json', 'README.md'}
    assert 'sdk: static' in (dest / 'README.md').read_text(encoding='utf-8')


def test_static_page_is_switched_to_static_mode_and_the_source_page_is_not(tmp_path):
    dest = tmp_path / 'out'
    dest.mkdir()
    push.assemble(dest, root=fake_root(tmp_path), kind='static')
    page = (dest / 'index.html').read_text(encoding='utf-8')
    assert push.STATIC_TAG in page and push.LIVE_TAG not in page
    assert push.LIVE_TAG in (ROOT / 'static' / 'index.html').read_text(encoding='utf-8')


def test_static_space_without_saved_answers_fails_loudly(tmp_path):
    root = fake_root(tmp_path)
    (root / 'static' / 'demo_cache.json').unlink()
    dest = tmp_path / 'out'
    dest.mkdir()
    with pytest.raises(SystemExit, match='demo_cache'):
        push.assemble(dest, root=root, kind='static')


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        push.assemble(tmp_path, root=fake_root(tmp_path), kind='nonsense')


def test_the_real_repo_has_saved_answers_to_publish():
    # CI runs from a checkout, so this catches a forgotten `git add static/demo_cache.json`.
    assert (ROOT / 'static' / 'demo_cache.json').exists()


def test_the_page_can_show_a_saved_answer_in_static_mode():
    page = (ROOT / 'static' / 'index.html').read_text(encoding='utf-8')
    assert "fetch('demo_cache.json')" in page and 'innerHTML' not in page


def test_docker_space_gets_the_whole_app(tmp_path):
    push.assemble(tmp_path, kind='docker')
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {'Dockerfile', 'pyproject.toml', 'src', 'static', 'README.md'}
    assert 'sdk: docker' in (tmp_path / 'README.md').read_text(encoding='utf-8')
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


def test_a_bad_space_kind_stops_the_upload(monkeypatch, capsys):
    monkeypatch.setenv('HF_TOKEN', 'x')
    monkeypatch.setenv('SPACE_REPO', 'me/space')
    monkeypatch.setenv('SPACE_KIND', 'gradio')
    assert push.main() == 1 and 'SPACE_KIND' in capsys.readouterr().out
