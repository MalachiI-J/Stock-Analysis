from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools import create_source_archive as archive_tool


def _write(path: Path, payload: bytes = b"content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_source_archive_includes_source_and_excludes_local_generated_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    included = {
        "main.py",
        "README.md",
        ".gitignore",
        ".env.example",
        "config/settings.yaml",
        "stock_scrapper/module.py",
        "tests/test_module.py",
        "tools/create_source_archive.py",
        "uv.lock",
    }
    for relative_path in included:
        _write(project / relative_path)

    excluded = {
        ".env",
        ".env.local",
        ".git/config",
        ".venv/Lib/site-packages/dependency.py",
        ".pytest_cache/state",
        ".agent_test_work/generated.py",
        "__pycache__/module.cpython-311.pyc",
        "data/market.db",
        "data/market.db-wal",
        "data/backups/market.db.bak",
        "data/raw/prices.parquet",
        "logs/run.log",
        "reports/summary.html",
        "stock_scrapper.egg-info/PKG-INFO",
        "build/package.py",
        "dist/old.zip",
        "temporary.tmp",
    }
    for relative_path in excluded:
        _write(project / relative_path)

    output = project / "dist" / "source.zip"
    result = archive_tool.create_source_archive(project, output)

    assert result == output.resolve()
    with zipfile.ZipFile(result) as archive:
        names = set(archive.namelist())
    assert included <= names
    assert names.isdisjoint(excluded)
    assert "dist/source.zip" not in names
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)


def test_source_archive_is_deterministic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(project / "main.py", b"print('hello')\n")
    _write(project / "config" / "settings.yaml", b"app_name: Test\n")

    first = archive_tool.create_source_archive(project, project / "dist" / "first.zip")
    second = archive_tool.create_source_archive(project, project / "dist" / "second.zip")

    assert first.read_bytes() == second.read_bytes()


def test_source_archive_preserves_existing_output_when_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    _write(project / "main.py")
    output = project / "dist" / "source.zip"
    _write(output, b"existing archive")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(archive_tool, "_write_archive", fail_write)

    with pytest.raises(OSError, match="simulated write failure"):
        archive_tool.create_source_archive(project, output)

    assert output.read_bytes() == b"existing archive"
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_source_archive_rejects_non_zip_output_without_overwriting_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(project / "main.py", b"source")

    with pytest.raises(ValueError, match=r"\.zip extension"):
        archive_tool.create_source_archive(project, project / "main.py")

    assert (project / "main.py").read_bytes() == b"source"
