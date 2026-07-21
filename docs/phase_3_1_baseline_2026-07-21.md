# Phase 3.1 baseline audit — 2026-07-21

This audit was captured before any Phase 3.1 schema or application changes.

## Repository and test baseline

- Existing repository: `Stock Scraper` (`stock_scrapper` package), clean Git worktree.
- Python project version: 0.3.0; database schema version: 3.
- Complete offline suite: **143 passed** in 6.23 seconds.
- `python main.py status`: 18,825 price rows, 5 collection runs, 6 analysis
  runs, 10 backtest runs, and 3 walk-forward runs.

## Database baseline

| Table | Rows |
|---|---:|
| analysis_runs | 6 |
| backtest_equity_curve | 3,267 |
| backtest_fills | 642 |
| backtest_metrics | 340 |
| backtest_orders | 642 |
| backtest_runs | 10 |
| backtest_signals | 40,176 |
| backtest_trades | 321 |
| collection_runs | 5 |
| data_quality_issues | 0 |
| market_regime_history | 6 |
| price_history | 18,825 |
| schema_metadata | 3 |
| stock_analysis | 21 |
| walk_forward_runs | 3 |
| walk_forward_windows | 6 |

All 15 symbols have exactly 1,255 rows from 2021-07-20 through 2026-07-20:
AAPL, AMZN, GLD, GOOGL, IWM, JPM, META, MSFT, NVDA, QQQ, SPY, TLT, TSLA,
WMT, and XOM.

The requested latest-five-row samples for SPY, AAPL, and TLT were inspected.
All end on 2026-07-20. AAPL's latest row was refreshed at
2026-07-21T12:39:20.958451+00:00 and has plausible completed-session volume.
SPY and TLT were collected on 2026-07-20 at approximately 11:22 and 12:28
America/New_York respectively, before the official close. Their July 20 rows
therefore cannot be treated as completed bars without reconciliation. SPY's
stored volume (12,286,193) is also far below the preceding completed session's
62,569,200, consistent with an intraday partial row.

Corporate-action fields are non-null in only 2,508 of 18,825 rows. There are 40
non-zero dividend rows and zero non-zero split rows. The database therefore
does not establish complete corporate-action coverage.

All 10 saved backtests use strategy `score_v1` version `1.0.0`; their dates,
hashes, and timestamps were inspected before migration. Existing deterministic
hash repetition across equivalent walk-forward windows is preserved as a useful
baseline, but stored runs do not yet have complete source provenance.

## Pre-migration plan

Schema version 4 will be applied transactionally and idempotently. It will:

1. Extend `price_history` with canonical bar lifecycle, exchange-close,
   collection timestamp, revision-count, and SHA-256 fingerprint metadata.
2. Add immutable `price_history_revisions` rows for material provider changes,
   with old/new fingerprints and values, changed fields, source, run, reason,
   and analysis-critical impact.
3. Add a separate `corporate_actions` table so action availability and source
   are explicit rather than inferred from nullable price columns.
4. Extend analysis/backtest run metadata with completed-session, warm-up,
   universe, software/strategy provenance, schema/source fingerprints, and
   requested/effective execution dates.
5. Add separate persisted strategy-diagnostic tables where needed; no future
   outcome will be part of signal generation.

Existing price rows and run records will not be deleted or recreated. Historical
rows strictly before the questionable latest session may be marked complete only
after OHLCV validation. Latest rows will initially remain `unknown` unless their
collection time proves they were collected after the exchange close plus provider
delay; reconciliation is responsible for replacing or confirming them. Any
failure rolls the migration back as one transaction.
