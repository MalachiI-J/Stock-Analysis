# Stock Scrapper

Stock Scrapper 0.3.0 is a free, local, explainable stock-market research and historical-backtesting application. It collects daily market data, preserves it in SQLite, calculates transparent technical evidence, saves reproducible analysis runs, and simulates a long-only strategy in one shared portfolio.

The project is educational research software. It does not provide personalized financial advice, place orders, connect to a brokerage, or guarantee investment performance.

## Phase 1 through Phase 3

### Phase 1: local market-data foundation

- Downloads daily OHLCV, adjusted close, dividends, and split data from yfinance.
- Uses the configured calendar-year lookback and incremental collection dates.
- Stores observations in `data/market.db` with one row per symbol and trading date.
- Validates price records and tracks data-quality issues without replacing missing values with zero.
- Supports local status, validation, logging, CSV output, and HTML output.

### Phase 2: explainable technical research

- Calculates trailing returns, Wilder RSI, true-range ATR, moving averages and slopes, time above moving averages, liquidity, volatility, downside risk, gap risk, drawdowns, and 52-week positioning.
- Aligns each symbol with SPY by trading date for benchmark-relative returns, beta, correlation, and relative-strength trend.
- Calculates one market context per analysis date using SPY, QQQ, IWM, and actual eligible-universe breadth.
- Produces separate measured-risk, technical-opportunity, and confidence scores.
- Applies explicit classification precedence and blocks scoring when critical information is unavailable.
- Uses SHA-256 issue fingerprints to deduplicate unresolved quality issues, resolve issues no longer detected, and reopen recurring issues.
- Saves analysis runs, exact component evidence, configuration hashes/snapshots, explanations, and market-regime history.
- Generates self-contained offline Phase 2 HTML and CSV reports with rankings, score changes, quality concerns, and inline adjusted-price/SMA charts.

### Phase 3: historical strategy validation

- Uses the same canonical analysis and eligibility logic as live Phase 2 research.
- Generates signals after session close and executes no earlier than the next available session's adjusted open.
- Simulates one shared long-only, unleveraged portfolio with cash, reserved cash, pending orders, positions, costs, and daily equity.
- Enforces position count, position weight, cash reserve, affordability, and configurable fractional-share rules.
- Supports equal-weight and optional volatility-adjusted sizing; it does not use Kelly sizing.
- Applies commission, adverse slippage, stop loss, trailing stop, maximum holding period, regime exits, and configurable final liquidation.
- Persists signals, rejected candidates, orders, fills, trades, equity, metrics, and walk-forward windows.
- Compares results with SPY buy-and-hold and cash.
- Produces deterministic offline reports and separate CSV logs for every persisted simulation.

SEC filings, FRED data, news, machine learning, brokerage connections, paper trading, real-money execution, intraday trading, short selling, leverage, options, and futures are outside the current scope.

## Windows and VSCode setup

Python 3.11 or newer is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

In VSCode, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. Run commands from the repository root so relative `config/`, `data/`, `logs/`, and `reports/` paths resolve consistently.

No API key or paid account is required. Runtime settings come from YAML; see [Configuration](#configuration).

## Command-line reference

### Collection, validation, and status

```powershell
# Incrementally collect missing daily rows for the configured watchlist
python main.py update

# Restrict collection or request the full configured lookback
python main.py update --symbols AAPL MSFT SPY
python main.py update --full-refresh

# Validate stored data, inspect local state, or run the daily workflow
python main.py validate
python main.py status
python main.py run
python main.py run --symbols AAPL MSFT --full-refresh
```

Collection is the network-dependent operation. Analysis, saved-result inspection, reporting, and backtesting use local SQLite data.

### Analysis and saved results

```powershell
# Current or historical analysis; --date is an alias for --as-of-date
python main.py analyze --symbols AAPL MSFT
python main.py analyze --symbols AAPL --as-of-date 2024-12-31
python main.py analyze --symbols AAPL --date 2024-12-31

# Read the latest saved analysis without creating a new run
python main.py scores
python main.py explain AAPL

# Explicitly calculate and save a new analysis before displaying it
python main.py scores --recalculate
python main.py explain AAPL --recalculate

# Inspect persisted runs
python main.py analysis-list
python main.py analysis-show --run-id <analysis-run-id>

# Build an offline Phase 2 report bounded by the requested date
python main.py report --symbols AAPL MSFT --date 2024-12-31
```

`scores` and `explain` are read-only by default. Recalculation occurs only when requested. Invalid dates, invalid configuration, missing data, partial failure, database failure, and complete failure return nonzero exit status rather than silently reporting success.

### Backtesting

```powershell
# Baseline score strategy and optional universe/range/cost overrides
python main.py backtest --strategy score_v1
python main.py backtest --strategy score_v1 --symbols AAPL MSFT SPY
python main.py backtest --start 2022-07-01 --end 2026-06-30
python main.py backtest --initial-cash 100000
python main.py backtest --commission-bps 1
python main.py backtest --slippage-bps 5

# A backtest does not download data unless this is explicitly supplied
python main.py backtest --strategy score_v1 --update

# Inspect and report a persisted simulation without rerunning it
python main.py backtest-list
python main.py backtest-show --run-id <backtest-run-id>
python main.py backtest-report --run-id <backtest-run-id>
python main.py backtest-compare --run-id <backtest-run-id>

# Evaluate fixed rolling development/validation/holdout windows
python main.py walk-forward --strategy score_v1
```

Backtest report generation reads the saved run and overwrites the same deterministic report paths; it does not duplicate the simulation or database rows.

### Clean source archive

```powershell
python tools/create_source_archive.py
```

The archive is written under `dist/`. It includes source, configuration, documentation, and tests while excluding Git metadata, environments, caches, compiled files, logs, reports, databases and backups, raw-data caches, egg-info, previous archives, and temporary files. It never deletes working files.

## Configuration

- `config/settings.yaml` controls local paths, data source, retry behavior, and historical lookback.
- `config/watchlist.csv` defines the static research universe.
- `config/scoring_rules.yaml` defines score weights, thresholds, regime settings, and scoring version.
- `config/backtesting_rules.yaml` defines strategy, portfolio, execution, cost, stop, benchmark, and walk-forward assumptions.

Configuration is validated before use. Unknown or missing scoring components are rejected, weights must be numeric and nonnegative, and each score's weights must total exactly 100. Configuration snapshots are canonicalized as sorted JSON and identified with stable SHA-256 hashes.

## Score definitions and classifications

All scores are deterministic 0–100 scales. An unavailable input remains unavailable; it is never silently treated as zero.

### Technical opportunity score

Higher means stronger price-based opportunity evidence. Its canonical components are:

- `long_term_trend`
- `multi_period_momentum`
- `relative_strength`
- `trend_quality`
- `volume_participation`
- `breakout_positioning`

There are no synthetic company-quality or valuation components. Those concepts require fundamental data and are not part of this technical score.

### Measured-risk score

Higher means more measured risk, not higher expected return. Evidence includes realized and downside volatility, drawdown, ATR and overnight gaps, beta, trend deterioration, liquidity, market regime, and data quality. Missing critical risk evidence can block scoring; noncritical missing evidence lowers confidence.

### Confidence score

Higher means the result is better supported and more complete. Confidence considers history completeness, freshness relative to the as-of date, unresolved quality issues, benchmark alignment, indicator availability, market-context availability, and agreement among trend, momentum, and relative strength.

### Classification precedence

The configurable classification rules are applied in this order:

1. `Data Blocked`
2. `Insufficient Data`
3. `High Risk`
4. `Avoid`
5. `Watch`
6. `Candidate`
7. `Strong Candidate`

A critical data issue overrides numerical scores. Market regimes are `Risk-On`, `Neutral`, `Risk-Off`, `Stress`, or `Insufficient Market Data`.

## As-of dates and no-lookahead design

Historical analysis is bounded at the database query, not merely labeled with an earlier filename. For an as-of date `T`:

- Stock, benchmark, market-context, breadth, and quality inputs are limited to dates on or before (T).
- `data_through_date` cannot exceed (T).
- Rolling indicators use trailing, non-centered windows.
- Future rows are not backfilled into missing historical sessions.
- Adding later stock, SPY, or watchlist rows does not change an earlier result.
- Phase 2 and Phase 3 use the same canonical feature, scoring, regime, classification, and eligibility logic.

Backtest timing is deliberately separated:

1. Session `T` closes.
2. Phase 2 evidence is calculated using information available through that close.
3. Candidates are ranked and orders are scheduled.
4. Orders execute no earlier than the next available session at adjusted open.
5. Commission and adverse slippage are applied to fills.

Signals are never executed at the close that generated them. Weekends and holidays are handled through the stored trading-session calendar. If the scheduled next session lacks a valid adjusted open, the order is rejected rather than deferred or filled with an invented price.

## Adjusted OHLC and corporate actions

Backtesting uses a consistent adjustment factor:

```text
adjustment factor = adjusted close / raw close
adjusted open      = raw open × adjustment factor
adjusted high      = raw high × adjustment factor
adjusted low       = raw low × adjustment factor
adjusted close     = reported adjusted close
```

Missing or invalid adjustment factors remain unavailable. Dividends are not counted a second time when adjusted prices already reflect them. Split and reverse-split handling preserves position continuity.

When a daily bar's high and low imply that competing stop/target events could both have happened, intraday order is unknowable. The default `adverse_first` ambiguity policy assumes the adverse event occurred first and records the ambiguity on the trade.

## Portfolio and `score_v1`

The simulator maintains one shared portfolio with cash, reserved cash, pending orders, long positions, average cost, realized and unrealized P&L, market value, equity, exposure, costs, and daily returns. It prohibits short selling, leverage, and negative cash.

The baseline `score_v1` entry rules use Candidate/Strong Candidate classification, configured opportunity/confidence/risk thresholds, allowed regimes, liquidity, and quality eligibility. When slots are limited, candidates are ranked deterministically by:

1. Higher opportunity
2. Higher confidence
3. Lower risk
4. Higher relative strength
5. Higher liquidity
6. Symbol

Exit reasons can include Avoid/High Risk classification, score deterioration, confidence loss, close below SMA200, Stress regime, stop loss, trailing stop, maximum holding period, or final liquidation. Exact entry and exit reasons remain attached to each persisted trade.

`rebalancing_frequency` selects the dates on which eligible new entries are reviewed and ranked. It does not force the sale or scheduled replacement of positions already held; configured exit rules determine when those positions close.

Position sizing supports equal weight and optional volatility adjustment. Missing volatility is not treated as low risk. Maximum positions, maximum position weight, cash reserve, fractional-share policy, affordability, commission basis points, minimum commission, and adverse slippage are all enforced from configuration.

## Metrics, benchmarks, and walk-forward validation

Performance reporting includes:

- Starting/ending equity, net profit, total return, CAGR, and annualized volatility
- Maximum drawdown, drawdown duration, Sharpe, Sortino, and Calmar ratios
- Exposure, turnover, trade count, win rate, average win, average loss, best trade, worst trade, profit factor, and expectancy
- Average holding period, consecutive wins/losses, commission cost, and slippage cost
- Monthly and annual returns
- SPY return/drawdown comparisons and cash comparison

Daily metrics use 252-session annualization unless configuration states otherwise. Undefined denominators produce unavailable metrics rather than misleading infinities or zeroes.

Walk-forward validation uses fixed warm-up and development periods as preceding context for rolling validation windows and one final holdout. The validation and holdout ranges are the periods actually simulated; development periods are recorded as fixed context, not run as separate optimization windows. The same immutable configuration is used throughout, so the workflow evaluates consistency across time without searching for or optimizing historical thresholds.

## Persistence and reports

SQLite is the system of record. Safe migrations preserve existing prices and add analysis, regime, backtest, trade, fill, equity, metric, and walk-forward tables with run identifiers, foreign keys, indexes, uniqueness rules, and transactional writes.

Phase 2 reports contain run metadata, as-of/data-through dates, score version/hash, regime evidence, candidate/risk rankings, components, factors, limitations, quality issues, prior-run changes, methodology, and inline adjusted-price/SMA20/SMA50/SMA200 charts.

Backtest reports contain assumptions, date/warm-up ranges, universe/exclusions, execution and cost rules, metrics and SPY comparison, inline equity/drawdown charts, period returns, complete trades/rejections, symbol/regime performance, and bias warnings. Separate CSVs cover summary, trades, all signals, rejected candidates, orders/fills, equity, monthly returns, and annual returns. Reports are self-contained and use no CDN.

## Project structure

```text
main.py                         CLI entry point
config/                         Settings, scoring, backtest rules, watchlist
stock_scrapper/analysis/        Indicators-to-score research workflow
stock_scrapper/backtesting/     Configuration, simulation, persistence, metrics, reports
stock_scrapper/collectors/      Daily market-data collection
stock_scrapper/migrations/      Safe SQLite schema migrations
stock_scrapper/processing/      Validation, indicators, relative strength
stock_scrapper/reporting/       Phase 2 offline reporting
tools/                          Clean source-archive tooling
data/                           Local SQLite and caches; not source-controlled
reports/                        Generated offline reports; not source-controlled
logs/                           Runtime logs; not source-controlled
tests/                          Offline deterministic pytest suite
```

## Testing

Tests are deterministic and do not require internet access.

```powershell
python -m pytest -q
```

## Limitations

- **Static watchlist:** the configured universe is not reconstructed historically.
- **Survivorship bias:** delisted, merged, bankrupt, or otherwise unavailable securities may be absent, which can overstate robustness.
- **Free-data limitations:** yfinance data may be delayed, revised, incomplete, rate-limited, or inconsistent across corporate actions.
- **Daily bars:** OHLC data cannot reveal the exact intraday order of events.
- **Historical simulation:** fills are modeled from stored bars and configured assumptions, not an exchange order book.
- **Research scope:** technical evidence omits fundamentals, macroeconomic releases, news, taxes, borrowing constraints, and individual circumstances.

## Financial and historical-results disclaimer

Stock Scrapper is educational research software, not a broker, investment adviser, fiduciary, or personalized recommendation service. Nothing produced by the application is an offer or instruction to buy or sell a security.

All scores, classifications, charts, comparisons, and backtests are hypothetical research outputs. Historical or simulated performance does not guarantee future results. Real trading can differ materially because of data revisions, liquidity, spreads, order priority, market impact, taxes, outages, corporate actions, and other factors. You are responsible for independent verification and any decisions you make.
