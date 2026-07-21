"""Create a clean, deterministic source archive for Stock Scrapper."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "backups",
        "build",
        "data",
        "dist",
        "htmlcov",
        "logs",
        "reports",
        "temp",
        "tmp",
        "venv",
    }
)

EXCLUDED_FILE_SUFFIXES = (
    ".7z",
    ".bak",
    ".backup",
    ".db",
    ".feather",
    ".gz",
    ".log",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyd",
    ".pyo",
    ".orig",
    ".rej",
    ".sqlite",
    ".sqlite3",
    ".so",
    ".swp",
    ".swo",
    ".tar",
    ".temp",
    ".tmp",
    ".whl",
    ".zip",
)

EXCLUDED_EXACT_FILE_NAMES = frozenset(
    {
        ".coverage",
        ".env",
        ".python-version",
        "coverage.xml",
        "thumbs.db",
        ".ds_store",
    }
)


def _is_excluded_directory(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in EXCLUDED_DIRECTORY_NAMES
        or lowered.endswith(".egg-info")
        or lowered.startswith(".agent_test")
        or lowered.startswith(".pytest-baseline-")
    )


def _is_excluded_file(path: Path) -> bool:
    name = path.name
    lowered = name.lower()
    if lowered in EXCLUDED_EXACT_FILE_NAMES or lowered.startswith(".coverage."):
        return True
    if lowered.startswith(".env.") and lowered != ".env.example":
        return True
    if lowered.endswith(EXCLUDED_FILE_SUFFIXES):
        return True
    if lowered.endswith(
        (".db-journal", ".db-shm", ".db-wal", ".sqlite-journal", ".sqlite-shm", ".sqlite-wal")
    ):
        return True
    return name.endswith("~") or (name.startswith("#") and name.endswith("#"))


def iter_source_files(source_root: Path) -> Iterable[Path]:
    """Yield eligible regular source files in deterministic relative-path order."""
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")

    eligible: list[Path] = []

    def _raise_walk_error(error: OSError) -> None:
        raise error

    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        current_path = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_excluded_directory(name) and not (current_path / name).is_symlink()
        )
        for file_name in sorted(file_names):
            candidate = current_path / file_name
            if candidate.is_symlink() or not candidate.is_file() or _is_excluded_file(candidate):
                continue
            eligible.append(candidate)

    yield from sorted(eligible, key=lambda item: item.relative_to(root).as_posix())


def _write_archive(temporary_path: Path, source_root: Path, source_files: Sequence[Path]) -> None:
    """Write files with stable ordering and metadata to a temporary ZIP archive."""
    root = source_root.resolve()
    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source_file in source_files:
            archive_name = source_file.relative_to(root).as_posix()
            archive_info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            archive_info.compress_type = zipfile.ZIP_DEFLATED
            archive_info.create_system = 3
            archive_info.external_attr = 0o100644 << 16
            archive.writestr(
                archive_info,
                source_file.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def create_source_archive(source_root: str | Path, output_path: str | Path) -> Path:
    """Create a clean archive and atomically replace the requested output path."""
    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError(f"Output path must use a .zip extension: {output}")
    source_files = tuple(iter_source_files(root))
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary_handle:
            temporary_name = temporary_handle.name
        temporary_path = Path(temporary_name)
        _write_archive(temporary_path, root, source_files)
        os.replace(temporary_path, output)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the source-archive CLI parser."""
    parser = argparse.ArgumentParser(description="Create a clean Stock Scrapper source archive")
    parser.add_argument("--source-root", type=Path, help="Project root; defaults to the parent of tools/")
    parser.add_argument("--output", type=Path, help="Output ZIP path; defaults to dist/ under the project root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archive CLI."""
    args = build_parser().parse_args(argv)
    project_root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or project_root / "dist" / f"stock_scrapper_source_{timestamp}.zip"
    try:
        archive_path = create_source_archive(project_root, output_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Unable to create source archive: {exc}", file=sys.stderr)
        return 1
    print(f"Created source archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
