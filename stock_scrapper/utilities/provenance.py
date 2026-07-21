"""Stable software and strategy provenance for reproducible research runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_scrapper.migrations.migration_manager import LATEST_SCHEMA_VERSION
from stock_scrapper import __version__


def source_fingerprint(base_dir: str | Path) -> str:
    root = Path(base_dir).resolve()
    paths = sorted(root.glob("stock_scrapper/**/*.py")) + sorted((root / "config").glob("*.yaml"))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    digest.update(f"schema:{LATEST_SCHEMA_VERSION}".encode())
    return digest.hexdigest()


def collect_provenance(base_dir: str | Path, strategy_name: str = "score_v1", strategy_version: str = "1.0.0", scoring_version: str | None = None) -> dict[str, Any]:
    root = Path(base_dir).resolve()
    app_version = __version__
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError): commit, dirty = None, None
    return {"application_version": app_version, "strategy_name": strategy_name, "strategy_version": strategy_version,
            "scoring_version": scoring_version, "schema_version": LATEST_SCHEMA_VERSION,
            "git_commit_hash": commit, "git_dirty": dirty, "source_fingerprint": source_fingerprint(root),
            "python_version": sys.version.split()[0], "platform_info": platform.platform()}
