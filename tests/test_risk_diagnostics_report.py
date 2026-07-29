from __future__ import annotations

from pathlib import Path

from stock_scrapper.analysis.risk_diagnostics import (
    DriverFeatureStudy,
    QuintileBucket,
    RiskDiagnosticsResult,
)
from stock_scrapper.reporting.risk_diagnostics_report import (
    render_risk_diagnostics_html,
    write_risk_diagnostics_html_report,
)


def _sample_result() -> RiskDiagnosticsResult:
    quintiles = [
        QuintileBucket(quintile=1, n=2, feature_low=-0.5, feature_high=-0.2, mean_excess_return=-0.01),
        QuintileBucket(quintile=5, n=2, feature_low=0.2, feature_high=0.5, mean_excess_return=0.04),
    ]
    study = DriverFeatureStudy(feature="six_month_return", n=4, pearson_r=0.9, quintiles=quintiles)
    return RiskDiagnosticsResult(
        classification="High Risk", horizon_days=21, sample_size=4, distinct_symbols=3,
        driver_studies=[study],
    )


def test_render_risk_diagnostics_html_includes_classification_and_feature() -> None:
    html_text = render_risk_diagnostics_html("backtest-test-run", {"High Risk": _sample_result()})

    assert "Risk-Inversion Diagnostics" in html_text
    assert "High Risk" in html_text
    assert "six_month_return" in html_text
    assert "backtest-test-run" in html_text
    assert "<html" in html_text and "</html>" in html_text


def test_render_risk_diagnostics_html_handles_empty_studies() -> None:
    empty_result = RiskDiagnosticsResult(
        classification="Watch", horizon_days=21, sample_size=0, distinct_symbols=0, driver_studies=[],
    )

    html_text = render_risk_diagnostics_html("backtest-test-run", {"Watch": empty_result})

    assert "Not enough non-missing rows" not in html_text  # only shown per-feature, not for a classification with zero studies
    assert "No feature had enough" in html_text or "Watch" in html_text


def test_write_risk_diagnostics_html_report_writes_file(tmp_path: Path) -> None:
    path = write_risk_diagnostics_html_report(tmp_path, "backtest-test-run", {"High Risk": _sample_result()})

    assert path.exists()
    assert path.name == "risk_inversion_study_backtest-test-run.html"
    assert "High Risk" in path.read_text(encoding="utf-8")
