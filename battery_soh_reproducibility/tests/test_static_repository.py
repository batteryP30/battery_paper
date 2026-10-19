from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_notebooks_in_release_tree():
    assert not list(ROOT.rglob("*.ipynb"))


def test_no_git_metadata():
    assert not (ROOT / ".git").exists()


def test_all_python_sources_compile():
    for path in ROOT.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_no_colab_or_user_specific_absolute_paths():
    forbidden = [
        "google" + ".colab",
        "/content" + "/drive",
        "C:" + "\\\\" + "Users" + "\\\\",
        "OneDrive" + " - University",
    ]
    for path in ROOT.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in {path}"


def test_no_generated_research_artifacts_are_packaged():
    prohibited_suffixes = {
        ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".pdf",
        ".pkl", ".joblib", ".zip", ".log",
    }
    offenders = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in prohibited_suffixes
    ]
    assert offenders == []
