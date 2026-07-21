"""Exact analysis-run report identity, manifests, and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from stock_scrapper.utilities.hashing import stable_sha256


def report_identity(scope: str, symbols: Sequence[str], run_id: str) -> str:
    short=run_id.rsplit("-",1)[-1][:8]
    if scope=="candidate_universe": label="candidates"
    elif scope=="all_data_symbols": label="all-data"
    else:
        joined="-".join(symbols)
        clean=re.sub(r"[^A-Za-z0-9-]+","-",joined).strip("-")
        label=f"custom-{clean}" if len(clean)<=48 else f"custom-{stable_sha256(list(symbols))[:10]}"
    return f"{label}_{short}"


def persist_report(conn: sqlite3.Connection, project_root: Path, run: Mapping[str, Any], paths: Mapping[str, Path]) -> Path:
    run_id=str(run["analysis_run_id"]); scope=str(run.get("analysis_scope") or "custom")
    version=int(conn.execute("SELECT COALESCE(MAX(report_version),0)+1 FROM analysis_reports WHERE analysis_run_id=?",(run_id,)).fetchone()[0])
    csv_path=Path(paths["csv"]); html_path=Path(paths["html"])
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    csv_hash,html_hash=digest(csv_path),digest(html_path)
    manifest_path=csv_path.with_suffix(".manifest.json")
    symbols=json.loads(run.get("analyzed_symbols_json") or "[]")
    manifest={"run_id":run_id,"as_of_date":run.get("as_of_date"),"scope":scope,"symbols":symbols,
      "application_version":run.get("application_version"),"scoring_version":run.get("scoring_version"),
      "configuration_hash":run.get("configuration_hash"),"data_hash":run.get("data_hash"),
      "source_fingerprint":run.get("source_fingerprint"),"report_hashes":{"csv_sha256":csv_hash,"html_sha256":html_hash}}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    relative=lambda p: p.resolve().relative_to(project_root.resolve()).as_posix()
    report_id=f"report-{run_id}-{version}"
    conn.execute("""INSERT INTO analysis_reports(report_id,analysis_run_id,generated_at,scope,csv_path,html_path,manifest_path,csv_sha256,html_sha256,report_version,status)
      VALUES(?,?,?,?,?,?,?,?,?,?, 'completed')""",(report_id,run_id,datetime.now(timezone.utc).isoformat(),scope,relative(csv_path),relative(html_path),relative(manifest_path),csv_hash,html_hash,version))
    conn.execute("UPDATE analysis_runs SET report_manifest_json=? WHERE analysis_run_id=?",(json.dumps(manifest,sort_keys=True),run_id))
    return manifest_path
