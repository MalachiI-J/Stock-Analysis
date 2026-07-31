# Stock Scrapper

Stock Scrapper 0.8.0 is a free, local, explainable stock-market research and historical-backtesting application. It collects daily market data, preserves it in SQLite, calculates transparent technical evidence, saves reproducible analysis runs, simulates a long-only strategy in one shared portfolio, tracks a user's real holdings against the same rules, screens a broader universe for new ideas, sizes advisory buy/sell recommendations within configurable risk limits, and offers a separate, clearly-labeled experimental statistical forecast alongside the deterministic score.

## Phase 6a trade recommendations and sizing (advisory only)

`recommend` is the first step toward the project's longer-term goal of an
automated trader with guardrails — deliberately built as a two-part process.
This part only recommends and sizes trades; it never places an order. It
combines two existing, independently-validated signals rather than inventing
new ones: SELL recommendations reuse `evaluate_holding()` from
`stock_scrapper/portfolio.py` (the exact rules a backtest would have exited
on), and BUY candidates are score_v1's own `Strong Candidate`/`Candidate`
classifications. Both experimental prediction models are attached to each BUY
as displayed context only — never as a gate, since neither's accuracy yet
warrants driving a real trade decision: `predict`'s probability
("model: 45% beats benchmark"), and `predict-v5`'s predicted excess-return
magnitude ("predict-v5: +12.3% predicted excess return"), including its
`[LOW CONFIDENCE]` suffix whenever that prediction is a statistically extreme,
likely-extrapolated outlier (see "Evaluation honesty" below) — the same
flag `predict-v5`'s own CLI output shows, now surfaced in the one command a
user actually checks daily instead of only in the standalone diagnostic run.

Position sizing follows the same weight-based approach as the backtester's
`BacktestConfig` (`max_position_weight`, `cash_reserve`), configured in
`config/trading_rules.yaml` alongside hard dollar caps
(`max_trade_dollar_amount`, `min_trade_dollar_amount`), a position-count cap
(`max_open_positions`), and a per-run cap on new buys
(`max_new_buys_per_run`). Available cash is derived, not stored: starting
capital minus every dollar ever committed to a lot, plus every dollar a sale
ever returned — no new database table was needed. `recommend` writes
`reports/recommendations_<date>.txt` plus a `.summary.json`, following the
same pattern as `digest`, and now runs as part of the daily automation loop
alongside it (see Daily automation below) — the toast notification includes
a buy/sell count with an explicit "advisory, unproven model" label so it's
never mistaken for a stronger signal than it is.

`trading_rules.yaml` also carries an `auto_execute` flag for a possible
future Part 3 (the program placing orders itself, opted into explicitly,
against a paper-trading broker first) — it is rejected as unsupported if set
to `true`. No broker integration exists yet; building one is a deliberate
later step, not a silent default.

`recommend-review` closes the accountability loop: the backtester and the
predictor's walk-forward folds both validate the underlying rules against
history, but neither tells you whether one specific past `recommend` run —
its specific candidates, its specific sizing, that specific day — actually
worked out. Given a saved `recommendations_<date>.summary.json`, it looks up
what each recommended symbol's price actually did between the recommendation
date and a later review date, grading BUYs exactly the way `predict` defines
success (beat the benchmark's own return) and SELLs in reverse (did the price
actually fall after being sold). One run is one data point, not a track
record — it's meant to be checked repeatedly over time as more dated
recommendation files accumulate, not read as a verdict from a single result.

## Phase 5 screening, notifications, and experimental prediction

`screen` scans `config/screening_universe.csv` — a static, illustrative list
of roughly 65 additional large-cap U.S. tickers, not an official index feed —
for symbols not already in the configured candidate universe, runs the same
Phase 2 analysis over them (as a non-persisted `custom`-scope batch, so
routine screening never clutters `analysis-list`), and reports any that
qualify as Candidate/Strong Candidate plus a full report for the rest. Pass
`--update` to explicitly collect data for the screening universe first.

`scripts/run_daily.ps1` shows the digest as a native Windows toast
notification (`scripts/send_toast.ps1`, using PowerShell's own registered
AppUserModelID rather than requiring a module install). This is why the
`\StockScrapper\DailyRun` scheduled task uses `LogonType=Interactive` — a
toast can only display in an active logged-in session. `digest` also writes
`reports/digest_<date>.summary.json`, a compact structured summary (counts,
top changes, symbols to consider selling) for exactly this kind of
non-terminal consumer, so the notification script never has to parse prose.
Email delivery was deliberately not built: it would need SMTP credentials,
which conflicts with this project's documented credential-free design
(see `.env.example`).

`predict` is an experimental statistical layer, fully separate from
`score_v1`: a small hand-rolled (not scikit-learn, to stay dependency-light
and fully transparent) logistic regression predicting the probability that a
symbol beats the benchmark's own return after `horizon_days` trading sessions
— not merely whether its own price rises. Raw direction is dominated by broad
market drift (in a sustained Risk-On stretch most stocks rise most of the time
regardless of anything stock-specific), so a model trained on raw direction
mostly just relearns "is the market going up," which it has no way to predict
better than chance; excess return over the benchmark isolates the
stock-picking signal the rest of this project is actually about. Features are
existing stored technical indicators plus two derived ones
(`market_regime_code`, that day's regime encoded as an ordinal, and
`opportunity_score_percentile`, the symbol's cross-sectional rank against the
rest of that day's analyzed universe). It is retrained from scratch on every
call — nothing is persisted — from historical (symbol, as-of-date) rows
sampled from stored price history, each paired with its own realized excess
return. A sample date only enters training if both the symbol's and the
benchmark's forward returns are fully computable from history already bounded
at or before the requested as-of date (see
`stock_scrapper/prediction/dataset.py`), so nothing about the future is ever
visible during training, matching the no-lookahead guarantee the rest of the
project depends on.

Performance is estimated via expanding-window walk-forward cross-validation,
split by **unique calendar date rather than row count**: rows are assembled
date-major (every eligible symbol for one sample date, then the next), so a
row-count boundary can and does land in the middle of one date's block,
silently putting some of that date's symbols in training and others in
testing for the same fold. Splitting by date first, then mapping back to
rows, guarantees no two rows sharing a date ever end up on opposite sides of
a fold boundary. Each fold additionally **purges** any training row whose
forward-return label resolves on or after the following test period's first
date — that row's *features* never see test-period data, but a label window
reaching into the test period is still informationally entangled with it,
and training on it would leak information about test-period outcomes.

Every fold reports its own training/test date ranges, distinct training/test
symbol counts, purged-row count, and its own **fold-specific baselines**: a
majority-class accuracy baseline and a constant-probability Brier baseline,
both computed from that fold's own test-period positive rate — not a fixed
50/50 assumption, and not the whole dataset's rate, which can differ
substantially from any one fold's (the CLI output shows each fold's own test
positive rate next to its accuracy for exactly this reason, rather than one
dataset-wide rate presented as if it applied to every fold). `holdout_accuracy`
and `holdout_brier_score` (and their baseline counterparts) are aggregated by
weighting each fold by its own test-sample count, not by naively averaging
fold-level numbers regardless of size. The model actually used for today's
predictions is then re-fit on the entire embargoed dataset, since a deployed
model should use all the history available to it. Fitting is deterministic
(zero initialization, full-batch gradient descent, no randomness), so the
same stored data always produces the same coefficients and predictions.
Configure via `config/prediction_rules.yaml` (horizon, lookback, feature
list, fold count, regularization, minimum sample count).

Every run is persisted to `prediction_runs`/`prediction_folds` (see
Persistence and reports) along with **evaluation provenance**: a fingerprint
of the assembled feature/label arrays, a hash of the symbol universe, a hash
of the configured feature list, a hash of the run configuration, and the
code's Git revision. This is what lets two runs be told apart honestly — a
rerun over identical data/symbols/features/config fingerprints identically
and is recognizable as a rerun, not a second independent piece of evidence;
only a run over genuinely new data (more history collected, a changed
universe, a changed feature set) produces a new fingerprint. See "Evaluation
honesty" below.

`predict-v4` is a second, separate experimental model answering a related but
distinct question: whether a *nonlinear* model does better than predict-v3's linear
one at the exact same task. It's motivated by a concrete finding, not a fishing
expedition — `investigate-risk-inversion` found a genuinely U-shaped relationship
between trailing six-month return and forward performance that a linear model's
single coefficient per feature structurally cannot represent. `predict-v4` fits a
hand-rolled (no scikit-learn, same reasoning as `predict`) gradient-boosted
regression tree ensemble (`stock_scrapper/prediction/gbm.py`) — deterministic,
exhaustive best-split search, no random row/feature subsampling — against a
**continuous** forward excess-return target (`build_regression_dataset`) rather than
predict-v3's binarized "beat the benchmark" label, keeping the magnitude information
the binary version throws away. It reuses the identical date-grouped, purged,
expanding-window walk-forward machinery as `predict` (same `_fold_boundaries`
splitting, same leakage purge), evaluated with regression-appropriate metrics
instead: mean squared error against each fold's own honest baseline (that fold's own
test-period target variance — the true minimum achievable MSE for always predicting
that fold's own mean), mean absolute error, and an information coefficient (Pearson
correlation between predicted and actual excess return). Runs persist to
`gbm_prediction_runs`/`gbm_prediction_folds` with the same evaluation provenance as
`prediction_runs`/`prediction_folds`. Configure via `config/prediction_rules.yaml`'s
`gbm:` section (tree count, depth, learning rate, leaf/split minimums, L2
regularization) — it shares `predict-v3`'s horizon/lookback/stride/folds/features so
the two models train and evaluate over identical rows.

The first real run against the full 20-year/25-symbol history
(`predict-v4-20260730193552-db691ae3`, 54,652 training samples) found the nonlinear
model does not help — if anything it is a clean, consistent negative result. Holdout
MSE (0.006157) is *worse* than the fold-specific baseline (0.005881), and it loses to
that trivial "always predict the fold's own mean" baseline in **every one of the 5
folds** individually (e.g. fold 4: 0.008480 vs. 0.008103; fold 5: 0.008358 vs.
0.008038) — a more consistent loss than predict-v3 shows against its own baseline.
The information coefficient is weak and inconsistent (-0.079 in the earliest fold,
then +0.01 to +0.05 in the other four; +0.0137 overall), far below anything that would
support real directional skill. Today's live predictions also surfaced a concrete
illustration of a known weakness of leaf-based models: one extreme outlier (INTC,
+18.50%, against a next-highest of +4.89% for every other symbol) — a sign of a
poorly-supported leaf estimate for a feature combination under-represented in
training, which a smooth linear model wouldn't produce in the same way. Matching the
hypothesis a nonlinear model would fix predict-v3's blind spot did not pan out in
practice; feature importances (led by `trend_slope_200`, `sixty_day_volatility`,
`beta`) are a genuinely different ranking from predict-v3's linear coefficients, but
that alone isn't evidence of anything actionable given the overall negative result.
See "Evaluation honesty" below.

`predict-v5` is a third, separate experimental model asking the one question the
other two couldn't: whether a genuinely new *data source* — not another reslice or
remodel of price/volume history — recovers an edge. `collect-fundamentals` pulls
point-in-time company financials directly from SEC EDGAR's free XBRL
`companyfacts` API (`stock_scrapper/collectors/sec_edgar_fundamentals.py`), storing
raw facts (net income, revenue, assets, liabilities, stockholders' equity, diluted
EPS, shares outstanding) each tagged with its own `filed_date` — the day a value
actually became public. `stock_scrapper/processing/fundamentals_features.py` turns
these into point-in-time features strictly bounded by that date (a fact filed after
an as-of date is never visible to it, mirroring price history's own no-lookahead
discipline): trailing P/E, price-to-book, debt-to-equity, revenue/earnings growth
(year-over-year, trailing-twelve-month), and return on equity. `predict-v5` reuses
`predict-v4`'s exact gradient-boosted model and evaluation methodology unchanged,
just with a wider `feature_keys` list (`config/prediction_rules.yaml`'s
`predict_v5.feature_keys`) — the base technical indicators plus the six
fundamentals above — so it stays directly comparable to predict-v4.

The first real run against the full candidate universe
(`predict-v5-20260731123026-dc54187d`, 23,400 training samples, 2010–2026 — a
narrower window than predict-v4's because a fundamentals-derived feature needs
several years of prior quarterly filings before it's usable) also came back
negative, and more decisively so: holdout MSE (0.008533) is worse than the
fold-specific baseline (0.007285) in **every one of the 5 folds** individually, and
the information coefficient (-0.0500 overall) is, if anything, more negative than
predict-v4's. What makes this result notable rather than a simple repeat: the two
fundamentals features (`trailing_pe`, `price_to_book`) were the model's *most*
important features by SSE-reduction gain (11.7% and 9.4%, ahead of every technical
indicator), with `earnings_growth_yoy` and `debt_to_equity` also in the top eight.
The model didn't ignore the new data — it leaned on it heavily — and still lost to
a trivial baseline in every fold. That is evidence the fundamentals themselves
don't carry real forward-return signal at this horizon (at least via this
point-in-time TTM approach), not evidence the model failed to look. A known
coverage gap: about a third of candidate symbols showed "unavailable" for today's
live prediction because their revenue figures use XBRL tag variants outside the
small alias list `sec_edgar_fundamentals.py` currently checks — a data-completeness
limitation, not a lookahead or correctness bug. See "Evaluation honesty" below.

**Does the horizon matter?** Fundamentals are classically a multi-quarter/multi-year
signal, not a ~1-month one — the 21-session horizon above was inherited from
predict-v3/v4 for comparability, not chosen because it suits fundamentals. Since
`--horizon-days` already lets any `predict-v5` run override that, two more runs
tested 63 sessions (~1 quarter) and 252 sessions (~1 year) against the same real
database, using existing code, not a new model. This initial scan (with the
15/25-symbol coverage and shared predict-v4 hyperparameters in place at the time —
see below for the fixes applied afterward) still established the basic pattern:

| Horizon | Holdout MSE vs. baseline | Folds beating baseline | Information coefficient | Top feature (share of gain) |
|---|---|---|---|---|
| 21 sessions | 0.008533 vs. 0.007285 (worse) | 0 of 5 | -0.0500 | `trailing_pe` (11.7%) |
| 63 sessions | 0.029014 vs. 0.025104 (worse) | 1 of 5 | +0.0113 | `sixty_day_volatility` (14.3%) |
| 252 sessions | 0.383058 vs. 0.349628 (worse) | 0 of 5 | +0.0410 | `return_on_equity` (29.9%) |

Fundamentals' share of feature-importance gain grows sharply with horizon — a
plausible pattern (profitability/quality factors are exactly what classic
long-horizon fundamentals literature would predict matters at this scale) — and the
252-session aggregate information coefficient is the highest of the three. But this
scan also surfaced a real, separate bug, not just a "coverage gap": about a third of
candidates showed "unavailable," and tracing why (rather than assuming it was the
already-known revenue-tag limitation) found that most of it was two concrete, fixable
problems — (1) large companies with noncontrolling interests (T, VZ, PG) tag only
`StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`, never plain
`StockholdersEquity`; (2) SEC's own ticker-to-CIK file maps "XOM" to an unrelated
shell entity ("ExxonMobil Holdings Corp") with zero XBRL facts, not the real Exxon
Mobil Corporation. Both are now fixed (a fallback alias, and a documented CIK
override, plus a safeguard that treats a zero-fact response as a collection failure
rather than a silent empty success), and a third case (companies that never tag
`Liabilities` directly, only `Assets` and `StockholdersEquity`) is handled by deriving
it from the accounting identity `Liabilities = Assets − StockholdersEquity` — exact,
not an approximation. Coverage went from 15/25 to 25/25 candidates.

With full coverage, the single pre-committed confirmatory retest at 252 sessions
(chosen before rerunning, not selected after seeing this run's result) is the most
interesting outcome of this entire investigation: training samples nearly doubled
(21,675 → 39,580), the overall MSE gap against baseline narrowed sharply (0.255943 vs.
0.254308 — about 0.6% worse, down from ~9.6% worse), the information coefficient rose
to **+0.1624** (the highest recorded anywhere in this project), and for the first time
**2 of 5 folds actually beat their own fold-specific baseline** on MSE (folds 3 and 5),
with 4 of 5 folds showing a positive information coefficient. All six fundamentals
features rank in the top 8 by gain, together accounting for nearly 70% of it.

This still falls short of a validated edge, and comes with a serious practical
caveat: today's live prediction for INTC was **+323.73%**, an obviously broken value
(predicting INTC will beat SPY by over 300 percentage points in a year is not a sane
forecast) — the same known tree-ensemble weakness already documented for
`predict-v4` (an outlier leaf estimate for a sparsely-populated feature combination),
worse here because fundamentals only update quarterly, so a 252-session horizon
effectively has very few independent fundamental "vectors" per symbol to split on.

Diagnosing this directly (not guessing) found the exact mechanism: INTC's current
`trailing_pe` is a large *negative* number (unprofitable trailing twelve months),
`earnings_growth_yoy` is -82.5%, and this combination is rare enough in 20 years of
training history that it likely lands in a leaf backed by only one or two genuinely
independent quarterly precedents — repeated across many trading days, so a leaf
"sample count" doesn't mean real statistical diversity here. `predict_v5.gbm` in
`config/prediction_rules.yaml` now carries its own (heavier) regularization than
predict-v4's shared `gbm` section — `min_samples_leaf`/`min_samples_split` raised
~4x, `l2_lambda` raised 10x, shallower/slower boosting — one clearly-justified
configuration change, not a hyperparameter scan. Rerunning with it (still the same
252-session horizon, full 25/25 coverage) genuinely improved the model: MSE gap
narrowed further and, for the first time anywhere in this project, **the aggregate
holdout MSE beats its baseline outright** (0.246875 vs. 0.254308), **3 of 5 folds**
now beat their own baseline (up from 2), and the information coefficient rose to
**+0.1624 → +0.1910**. But INTC's outlier barely moved (+323.73% → +266.11%, despite
4x/10x stronger regularization) — evidence this reflects a genuinely rare historical
precedent the model is extrapolating from, not a simple overfitting artifact
regularization alone fixes.

Rather than keep chasing the outlier itself, `predict_v5`/`predict_v4`'s
`SymbolExcessReturnPrediction` now carries a `low_confidence` flag: any live
prediction more than `OUTLIER_STD_MULTIPLIER` (3) training-target standard
deviations from zero is labeled `[LOW CONFIDENCE: ...]` in the CLI output rather
than shown with the same apparent trustworthiness as an ordinary prediction — the
raw value is never suppressed or altered, only flagged (`gbm_service.py`). On the
real run above, INTC (+266.11%) is flagged; every other symbol, including TSLA's
also-large +44.24%, is not.

`predict-v5`'s prediction (and its `low_confidence` flag) is now also attached to
`recommend`'s BUY output as displayed context (see "Phase 6a" above) — the same rule
as `predict`'s probability: never a gate, always labeled, so the outlier-flagging
mechanism protects the one command actually checked day-to-day, not only the
standalone diagnostic run.

**Read together, this is the strongest result in the whole project** — real,
diagnosed, and honestly improved aggregate statistics — **but still not a validated
edge**, and the live model needs its outlier-flagging mechanism (now in place) before
any of its predictions should be trusted at all. Before doing anything else with
predict-v5, the honest next step is a genuinely new out-of-sample confirmatory run
(e.g. after the next quarter's fundamentals update) rather than another rerun of the
same 2010–2026 data.

`validate-signals` asks a complementary, higher-power question about
`score_v1` itself: across every day a symbol was ever classified —
independent of whether the portfolio backtester had room to act on it, which
makes this a different and more statistically powerful check than the
backtest's own trade-level diagnostics — did that classification's forward
performance actually differ from the rest? It re-runs a fresh full-history
backtest (so classifications reflect the current scoring code, not a
possibly-stale prior run), buckets every classified instance by
classification, and reports each bucket's hit-rate, mean/median forward
excess return against the benchmark, distinct-symbol count, a naive and a
Bonferroni-corrected p-value, and a symbol-weighted mean with a 95%
confidence interval. The p-values are explicitly labeled naive/descriptive,
not validated inferential statistics: with `horizon_days`-long overlapping
lookaheads and rows clustered by symbol, a bucket's row-count is not a count
of independent trials, and the Bonferroni correction only accounts for
testing all five buckets rather than one, not for that deeper
non-independence. Buckets backed by fewer than four distinct symbols are
flagged as concentration-prone — their apparent effect can be one or two
stocks' history repeated many times across overlapping daily windows, not a
population-level pattern. This caught a real instance in this project's own
history: an apparently significant "High Risk" outperformance turned out to
be exactly two symbols (out of seventeen) mid-rally, not a broad effect.
A monotonicity check (does mean excess return actually rank Strong Candidate
> Candidate > Watch > Avoid > High Risk) is reported both day-weighted and
symbol-weighted, since the two can diverge when a bucket is concentrated.

### Evaluation honesty: what counts as evidence

Both `predict` and `validate-signals` are built around one recurring
distinction that is easy to blur and important to keep straight:

- **A raw row** is one (symbol, date) observation. Thousands of rows sound
  like a lot of evidence, but rows are not automatically independent.
- **A distinct symbol** is the more defensible unit of independence when
  rows overlap in time or share a company — a bucket or fold backed by two
  stocks is two data points, not hundreds, no matter how many daily rows
  those two stocks contributed.
- **An independent evaluation block** (a walk-forward fold, a validation
  window) is a chronological slice evaluated once, without being used to
  tune anything. More non-overlapping blocks that agree is real evidence;
  one block, however large, is one data point.
- **A rerun over identical data** — same dataset fingerprint, same symbol
  universe, same feature set, same configuration — is not new evidence,
  even if it is run again next week. It confirms determinism, nothing more.
- **Genuinely new out-of-sample evidence** requires new data: more history
  collected since the last run, so the fingerprint changes and a fold or
  window covers dates never evaluated before.

The candidate universe was deliberately widened from an original 10 mega-cap,
mostly tech-weighted names to 25 spanning healthcare, staples, media,
telecom, industrials, auto, financials, energy, and utilities — including
several names with genuine multi-year drawdowns (INTC, PYPL, BA, T, VZ)
rather than only names that happened to ride a single tech/AI bull market.
`historical_lookback_years` was then extended from 5 to 20, which matters
independently of universe width: with ~5 years of history the portfolio
`walk-forward` command could only fit one validation window plus one
holdout, so a single window's outcome — good or bad — was never enough to
trust on its own. With 20 years it fits 15 validation windows plus one
holdout, a roughly 15x increase in independent evaluation blocks (see the
distinction above between one block and many).

The wider, deeper evidence base now gives a consistent, and mostly negative,
answer for both signals:

- **`score_v1` portfolio walk-forward** (15 validation windows + 1 holdout,
  2009–2026): the strategy beat SPY in only 5 of 15 validation windows
  (33%) and lost the holdout outright (most recent 12 months: active return
  −11.95%, Sharpe 0.47 vs. benchmark Sharpe 1.35). No consistent edge across
  independent periods.
- **`score_v1` classification hit-rate** (`validate-signals`, 109,706
  classified rows, 20–25 distinct symbols per bucket — resolving the
  concentration problem described above): "Strong Candidate" shows a real
  but thin edge (mean excess return +0.99%; symbol-weighted 95% CI
  [+0.01%, +1.42%], barely excluding zero). Monotonicity still fails, and
  now more seriously than before — "High Risk" posts the *largest* excess
  return of any bucket (mean +3.24%; symbol-weighted CI [+1.82%, +4.48%],
  clearly excluding zero, backed by 20 distinct symbols), so this is no
  longer a two-symbol concentration artifact but a genuine, well-supported
  inversion in the risk classification. It is reported here as an honest
  anomaly, not evidence of a usable signal — it could reflect a real risk
  premium (riskier names carry higher expected return) or a miscalibration
  in how risk is scored; distinguishing those would require dedicated
  investigation, not a reinterpretation of this data. The daily Phase 2
  report now surfaces this same Strong Candidate / High Risk figure (dated
  to the latest `validate-signals` artifact on disk) directly above the
  matching ranking table, with the same "descriptive anomaly, not a live
  prediction" framing — see "Persistence and reports" above.
- **`predict-v3`** (54,627 samples, 2008–2026, 5 purged walk-forward folds):
  holdout accuracy 50.6% against a sample-weighted baseline of 51.6%, and
  Brier score 0.2504 against a baseline of 0.2497 — below baseline on both
  metrics overall. Only 1 of 5 folds beats its own fold-specific baseline on
  accuracy; the most recent fold (2023–2026) is the worst of all five.
- **`predict-v4`** (54,652 samples, same window, gradient-boosted regression on
  continuous excess return): holdout MSE 0.006157 against a fold-specific
  baseline of 0.005881 — worse than baseline overall, and worse in all 5 of 5
  folds individually, a more consistent loss than predict-v3's. Information
  coefficient is weak and inconsistent (-0.079 to +0.05 across folds). Moving
  to a nonlinear model did not recover the edge a linear model might be
  missing.
- **`signal-capture-test`** (same 15 validation windows + holdout as the
  portfolio walk-forward, `score_v1`'s own trading rules stripped out): beat
  SPY in only 2 of 15 validation windows (13%, versus `score_v1`'s own 5 of
  15) and lost the holdout too. Removing stop-loss/trailing-stop/regime-gate/
  early-exit rules made results *worse*, not better — evidence those rules
  are doing real protective work rather than suppressing a hidden edge.
- **`predict-v5`** (23,400 samples, 2010–2026 — narrower than predict-v4's
  window because point-in-time fundamentals require several years of prior
  quarterly filings before a sample date has a usable trailing-twelve-month
  figure): predict-v4's identical gradient-boosted model and evaluation
  methodology, widened with point-in-time SEC EDGAR fundamentals (trailing
  P/E, price-to-book, debt-to-equity, revenue/earnings growth, return on
  equity — see "Fundamentals data" below). Holdout MSE 0.008533 against a
  fold-specific baseline of 0.007285 — worse than baseline overall, and worse
  in all 5 of 5 folds individually, the same clean negative pattern as
  predict-v4. Information coefficient is weak and, on balance, more negative
  than predict-v4's (-0.0500 overall; per-fold range -0.1505 to +0.0083).
  Notably, this is *not* a case of the model ignoring the new features:
  `trailing_pe` and `price_to_book` were its two most important features by
  SSE-reduction gain (11.7% and 9.4%, ahead of every technical indicator), and
  `earnings_growth_yoy`/`debt_to_equity` also ranked in the top eight. The
  fundamentals were used heavily and still didn't produce a validated edge —
  evidence against the data itself carrying real forward-return signal at this
  horizon, not evidence the model failed to look. This closes off the last
  remaining untried lever (a genuinely new data source) from this angle.

Taken together, neither signal has demonstrated a validated edge, and the
deeper, wider sample — plus three further, more targeted attempts to find one
(a nonlinear model, a trading-rules-stripped direct capture of the
classification edge, and a genuinely new fundamentals data source) — makes
that conclusion more solid than the earlier narrower one did. This is not a
case where more data, a different model, or a new data source revealed a
hidden edge; every additional check has removed ambiguity in the negative
direction. This is stated plainly rather than glossed over. A future claim of
validated edge would require a model or ranking to beat its own fold/window-
specific baseline consistently across multiple non-overlapping periods, not on
a rerun of the same data. See Limitations.

### Investigating the High Risk inversion

`validate-signals` describes the "High Risk" inversion honestly but does not explain
it: why does the classification bucket meant to flag the riskiest names post the
*largest* forward excess return, on a symbol-diverse sample that rules out the old
concentration-artifact explanation? `python main.py investigate-risk-inversion` follows
up with a narrower, exploratory question — within one classification bucket, does the
forward excess return concentrate in a specific range of an already-computed technical
indicator (no new feature engineering; the same fields `predict-v3` and
`risk_score.py`'s components already use)? Rows are split into 5 equal-sized quintiles
per indicator, and each quintile's mean forward excess return is reported alongside a
Pearson correlation across the whole bucket, plus a self-contained HTML report
(`reports/risk_inversion_study_<run_id>.html`) with a diverging bar chart per indicator.

Two candidate explanations motivated the indicator list: **mean reversion** (a
concentration in the most-negative quintile of trailing-return/drawdown fields would
suggest a bounce-back effect) versus **risk premium** (a pattern that scales with
volatility/beta magnitude regardless of trailing-return sign would suggest compensated
risk-taking instead). This is exploratory hypothesis generation, not a validated
result — the same overlapping-window, symbol-clustered non-independence documented
under "Evaluation honesty" applies here too, and a quintile split across several
correlated indicators can look suggestive by chance. It changes nothing about
score_v1's scoring, thresholds, or classification logic.

The first real run against the full 20-year/25-symbol history
(`backtest-20260729182530-b6a3a71d`) found the "High Risk" effect is not spread evenly
across its 4,685 classified rows (20 symbols) — it concentrates in two places. The
bottom quintile of trailing `six_month_return` (-90% to -47%) posts a mean forward
excess return of +8.33%, far above every other quintile of that feature (which cluster
near 0-4%) — a pattern consistent with a sharp bounce back from a severe decline rather
than a straight-line relationship (the whole-bucket Pearson correlation is near zero,
-0.038, precisely because the effect sits in one tail, not along a line).
`one_year_max_drawdown` is the strongest and most linear relationship found (r=+0.150):
rows with the deepest trailing one-year drawdown (top quintile) posted +8.41% mean
forward excess return versus +1.65% for the shallowest-drawdown quintile, and
`atr_percentage` shows a similar, weaker version of the same shape (r=+0.119). Plain
volatility and beta magnitude (`twenty_day_volatility` r=+0.002, `beta` r=-0.057) show
essentially no relationship. Read together, this leans toward a
mean-reversion-after-severe-decline story more than a clean compensated-risk-premium
one — but it remains exploratory and descriptive, not a validated result, and does not
by itself suggest anything score_v1 or predict-v3 currently exploits.

"Strong Candidate"'s much thinner overall edge (see above) is not spread evenly either:
every studied feature's top quintile (highest trailing momentum/return, volatility, or
drawdown magnitude) posts roughly +2.3% to +2.6% mean forward excess return versus
+0.2% to +0.7% in the bottom quintile, with modest positive correlations throughout
(+0.08 to +0.15) — the bucket's aggregate edge concentrates among its
higher-momentum, higher-volatility members, not across every "Strong Candidate" row
equally. Full quintile tables and correlations for every studied indicator, both
buckets, are in `reports/risk_inversion_study_backtest-20260729182530-b6a3a71d.json`
and its paired HTML report. Neither pattern changes score_v1 or predict-v3's current
behavior; both are recorded as a starting point for a future, more targeted hypothesis
or feature, not as anything actionable today.

### Can the "Strong Candidate" edge be captured directly?

`validate-signals` found "Strong Candidate" has a thin but real classification-level
edge (symbol-weighted mean excess return +0.99%, 95% CI [+0.01%, +1.42%] — barely
excluding zero), yet the full `score_v1` portfolio backtest still loses to SPY in most
walk-forward windows. That gap raises an obvious question: is the edge real but being
destroyed by `score_v1`'s own trading rules (entry/exit thresholds, stop-loss,
trailing-stop, regime gating), or does the edge simply not survive contact with a real
holding period and real costs? `python main.py signal-capture-test` isolates the
question directly: entry is gated on the "Strong Candidate" classification alone (no
extra opportunity/volume/confidence/risk/regime filters), every early-exit path
(stop-loss, trailing-stop, profit-target, classification-based exit, regime-based exit)
is disabled, and the only way out of a position is an unconditional exit after exactly
`horizon_days` sessions — the same horizon `validate-signals` measures against. Realistic
commission/slippage still apply, and position limits are widened (25 concurrent
positions) so the cap itself can't be the bottleneck. This reuses the exact same
purged, date-grouped walk-forward machinery as `walk-forward`, so results are directly
comparable window-for-window.

The answer, on the full 20-year/25-symbol history (`wf-signalcapture-20260730190245-2d67f6c0`):
**worse**, not better. This maximally-permissive, risk-control-free variant beat SPY in
only 2 of 15 validation windows (13%) and lost the holdout too (active_return -4.78%),
compared to `score_v1`'s own 5 of 15 (33%) with its full rule set intact. Removing the
stop-loss/trailing-stop/regime-gate/early-exit rules did not unlock the classification's
thin edge — it made outcomes worse, which is the opposite of what "the trading rules
are competing with a real edge" would predict. The more consistent explanation:
`score_v1`'s exit rules are doing real protective work (cutting losing positions before
a full `horizon_days` of downside), and the thin average edge `validate-signals` measures
across *all* classified instances unconditionally does not, by itself, translate into a
strategy worth running without risk management. This is exploratory diagnostic evidence,
not a proposed change to `score_v1` — it changes nothing about its scoring, thresholds,
or classification logic — but it closes off "just remove the trading rules" as a path to
a validated edge.

## Phase 4 real-portfolio tracking

`portfolio-buy` and `portfolio-sell` record real owned lots and closing sales
in SQLite; `portfolio-sell` closes the oldest open lots first (FIFO) and
rejects selling more shares than are currently held rather than partially
filling. `portfolio-show` lists open positions with current value, unrealized
P&L, and a rules-based hold/sell recommendation; add `--closed` for realized
lots and P&L.

A held position's recommendation reuses `evaluate_rule_based_exit` from
`stock_scrapper/backtesting/exit_rules.py` — the exact classification/regime/
score/SMA200/holding-period exit logic `score_v1` uses to close a backtested
position — plus a close-price stop-loss/trailing-stop check against the
position's average cost basis and highest close since it was opened. A held
symbol outside the analyzed candidate universe still gets the price-based
check; it just has no classification-based signal since nothing scores it.
`update` and `run` always additionally collect price data for every symbol
with an open lot, even if it is outside the configured watchlist, so a real
holding's evaluation has current data.

`digest` includes a "YOUR HOLDINGS" section built from the same assessment.
Recommendations are rule-based research output, not investment advice, and a
missing current price is reported as unavailable rather than assumed.

`portfolio-compare` reports real realized/unrealized P&L against a SPY
"shadow portfolio": the same dollars, invested in the benchmark on the same
days each lot was opened or sold, using each sale's own recorded `lot_id` to
match it to its true entry date rather than an average. A lot only counts
toward the shadow total if every benchmark price point it needs is available,
so the reported shadow total and its invested-capital denominator always
match; excluded lots and symbols with no current price are listed separately
rather than silently folded into the totals. This is a fairer comparison for
irregular real-world trade timing than a naive total-cost-vs-current-value
ratio, but it still ignores taxes, fees, and dividends.

`cleanup-logs` deletes files under `logs/` older than `logs_retention_days`
(default 30; override with `--days`). By default it never touches `reports/`,
since Phase 2/3 report files can be referenced by persisted
`analysis_reports`/`backtest_runs` rows and deleting them without removing
those rows would orphan a database reference. `--include-reports` additionally
deletes old digest (`.txt`/`.summary.json`), data-health, and screener
(`stock_summary_*_screen-*`) files specifically — the only report outputs
never referenced by any saved-run row, identified by filename pattern, never
the canonical/custom/backtest ones. `scripts/run_daily.ps1` runs
`cleanup-logs --include-reports` automatically after each daily digest.

## Phase 3.2 calibration and diagnostics

Revision policy `revision-v2` uses configurable absolute and relative tolerances.
Exact differences are retained as evidence, while sub-tolerance floating-point
changes are classified as `precision_noise`, do not increment material revision
counts, and do not overwrite the stable stored value. `revisions-classify`
classifies retained legacy audit rows without deleting them.

`corporate-actions-refresh --full` records the complete period checked for every
symbol, including successful responses containing no actions. Data health now
checks expected XNYS sessions, non-session dates, action coverage, adjustment
factors, revision materiality, freshness, invalid bars, and unresolved issues.

Backtests enforce the configured `reject`, `shift_start`, or
`allow_with_warning` warm-up policy using completed benchmark sessions. New runs
persist requested/effective dates, warm-up evidence, universe, health, action
coverage, revision policy, and software provenance. Strategy `score_v1` is now
version 1.1.0 because warm-up behavior affects simulation eligibility.

Benchmark diagnostics include CAGR, volatility, Sharpe, Sortino, Calmar,
tracking error, information ratio, capture ratios, beta, and correlation from
the same effective dates as the strategy. Symbol attribution, concentration,
cash/exposure evidence, forward signal outcomes, and exit diagnostics are
calculated only after the simulation and cannot influence its decisions.

The project is educational research software. It does not provide personalized financial advice, place orders, connect to a brokerage, or guarantee investment performance.

## Phase 3.1 market-data integrity

Daily bars use the official `XNYS` calendar in `America/New_York`. A bar is
complete only after the official close (including early closes), the configured
provider delay, and OHLCV validation. Incomplete, invalid, and unreconciled
latest bars are excluded from normal analysis and all backtests.
`analyze --include-incomplete-bars` is an explicitly warned diagnostic override.

Updates revisit a configurable recent-session overlap and compare stable SHA-256
row fingerprints. Changes create immutable `price_history_revisions` audit rows;
identical rows are untouched. Use `reconcile-prices --sessions 30` for a recent
repair or `reconcile-prices --full` for an intentional full refresh. History is
never silently rewritten. Explicit actions are stored in `corporate_actions`
when supplied. yfinance can revise adjusted prices or omit actions, so missing
actions mean unavailable—not proof no action occurred. Backtests use adjusted
OHLC and do not separately credit dividends, avoiding double counting.

Candidate, benchmark, market-context, and defensive roles are separate. Only
candidates trade; `universe-validate` warns about benchmark/candidate overlap.
Persisted runs carry configuration, data, deterministic-result, source-code,
application, strategy, Git, Python, platform, and schema provenance. Increment
the strategy version whenever entry/exit, ranking, sizing, costs, stops, regime,
score, or classification behavior changes.

```powershell
python main.py market-session
python main.py data-health
python main.py data-health-report
python main.py reconcile-prices --sessions 30
python main.py revisions --symbol AAPL
python main.py corporate-actions --symbol AAPL
python main.py universe-show
python main.py provenance
```

Critical market-data health blocks normal live classifications. Benchmark and
strategy comparisons must use the same effective session and adjusted-price
basis. Counterfactual diagnostics are research aids, not automatic optimization;
strategy underperformance must remain visible. Historical performance does not
guarantee future results.

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

## Daily automation

`scripts/run_daily.ps1` runs `python main.py run` (collect → validate →
analyze → report), `python main.py digest`, and `python main.py recommend`,
in that order, logging combined output to `logs/daily_run_<timestamp>.log`.
The Windows toast notification combines the digest and recommendation
summaries into one message, with the recommendation line explicitly marked
"advisory, unproven model" so it never reads as a stronger signal than it
is. After the notification, the script opens that day's canonical Phase 2
HTML report (`stock_summary_<date>_candidates_<hash>.html`, located by its
documented filename pattern) in the default browser, gated by
`open_reports_automatically` in `config/settings.yaml` (default `true`); if
no report was produced for today, this is skipped and noted in the log
rather than treated as a failure. It uses `cmd.exe` for output redirection rather than
PowerShell's native `2>&1`/`*>>`, which otherwise wraps every stderr line
from a Python process (the app logger writes `INFO` to stderr) in a
spurious `NativeCommandError` and can emit UTF-16 log files.

A Windows Task Scheduler task (`\StockScrapper\DailyRun`) runs this script
Monday–Friday. Inspect or change it with:

```powershell
Get-ScheduledTask -TaskPath "\StockScrapper\" -TaskName "DailyRun"
Start-ScheduledTask -TaskPath "\StockScrapper\" -TaskName "DailyRun"   # run once now
Disable-ScheduledTask -TaskPath "\StockScrapper\" -TaskName "DailyRun"
Unregister-ScheduledTask -TaskPath "\StockScrapper\" -TaskName "DailyRun"
```

The scheduled time is a plain local-clock trigger, not timezone-aware; it
assumes the machine's clock matches `market_data.timezone` in
`config/settings.yaml` (`America/New_York`). The pipeline itself is
correct regardless of exactly when it runs, since collection and analysis
are bounded by the official XNYS calendar and the configured provider
delay — running earlier just means more recently completed sessions may
not yet be available, and `data-health`/`market-session` remain the way to
check what is currently available.

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

# Plain-language daily buy/watch/sell digest from the latest saved run
python main.py digest
python main.py digest --recalculate
python main.py digest --run-id <analysis-run-id> --no-save

# Record real buys/sells and inspect your actual holdings
python main.py portfolio-buy --symbol AAPL --shares 10 --price 150.25 --date 2026-01-05
python main.py portfolio-sell --symbol AAPL --shares 4 --price 165.00 --date 2026-03-01
python main.py portfolio-show
python main.py portfolio-show --symbol AAPL --closed

# Compare your real P&L with a same-dollars-same-days SPY shadow portfolio
python main.py portfolio-compare
python main.py portfolio-compare --symbol AAPL --as-of-date 2026-06-30

# Sized buy/sell recommendations from the latest saved run and your real holdings (advisory only)
python main.py recommend
python main.py recommend --recalculate
python main.py recommend --run-id <analysis-run-id> --no-save

# Check how a past recommend run's suggestions actually performed
python main.py recommend-review --recommendation-date 2026-07-27
python main.py recommend-review --recommendation-date 2026-07-27 --as-of-date 2026-08-17

# Delete old log files; --include-reports also removes old digest/data-health/screener files
python main.py cleanup-logs
python main.py cleanup-logs --days 7
python main.py cleanup-logs --include-reports

# Scan a broader static universe for new Candidate/Strong Candidate ideas
python main.py screen
python main.py screen --update
python main.py screen --universe-path config/screening_universe.csv

# EXPERIMENTAL: statistical forward-return prediction, separate from score_v1
python main.py predict
python main.py predict --symbols AAPL MSFT --as-of-date 2026-06-30

# Does score_v1's classification beat the benchmark forward, independent of
# the portfolio backtester's own sizing/timing? See "Evaluation honesty" above.
python main.py validate-signals
python main.py validate-signals --start 2022-07-20 --end 2026-06-30 --horizon-days 21

# EXPLORATORY: which already-computed technical indicators drive a classification
# bucket's forward excess return? See "Investigating the High Risk inversion" below.
python main.py investigate-risk-inversion
python main.py investigate-risk-inversion --classifications "High Risk" "Strong Candidate"

# EXPLORATORY: is validate-signals' Strong Candidate edge capturable once score_v1's own
# trading rules stop competing with it? See "Can the Strong Candidate edge be captured
# directly?" above.
python main.py signal-capture-test
python main.py signal-capture-test --horizon-days 21

# EXPERIMENTAL: gradient-boosted regression on continuous excess return, separate from
# score_v1 and predict-v3. See "Phase 5" above.
python main.py predict-v4
python main.py predict-v4 --symbols AAPL MSFT --as-of-date 2026-06-30

# Collect point-in-time SEC EDGAR fundamentals for the candidate universe (manual/
# periodic -- not part of daily `run`; fundamentals change quarterly, not daily).
python main.py collect-fundamentals
python main.py collect-fundamentals --symbols AAPL MSFT

# EXPERIMENTAL: predict-v4 widened with point-in-time fundamentals. See "Phase 5" above.
python main.py predict-v5
python main.py predict-v5 --symbols AAPL MSFT --as-of-date 2026-06-30
```

`digest` reads the same saved classifications as `scores`/`explain` and groups
them into BUY (Candidate/Strong Candidate), SELL/AVOID-if-held (Avoid/High
Risk), WATCH, and DATA ISSUES, plus a change log against the previous saved
run. It writes `reports/digest_<as-of-date>.txt` unless `--no-save` is given.
It is a rendering of existing scores, not a new calculation, and carries the
same research-only disclaimer as every other report.

`scores` and `explain` are read-only by default. Recalculation occurs only when requested. Invalid dates, invalid configuration, missing data, partial failure, database failure, and complete failure return nonzero exit status rather than silently reporting success.

### Universe-aware analyses and canonical runs

The configured **candidate universe** is the 25 stocks eligible for analysis and trading — deliberately spanning multiple sectors and market histories rather than a single mega-cap/tech cohort (see "Evaluation honesty" above). The **data universe** is the ordered union of candidates, SPY, market context (SPY/QQQ/IWM), and defensive context (TLT/GLD), 30 symbols in total. Data collection, reconciliation, validation, and health commands default to all 30 data symbols. Analysis, reporting, backtesting, and walk-forward commands default to the 25 candidates; context assets are still loaded internally for relative strength, beta, breadth, correlation, and regime calculations.

```powershell
# Analyze and save the configured candidates as the canonical daily run
python main.py analyze
python main.py scores
python main.py report

# Deliberate alternatives
python main.py analyze --scope all-data
python main.py analyze --symbols AAPL MSFT
python main.py scores --latest-any
python main.py scores --run-id <analysis-run-id>
python main.py report --run-id <analysis-run-id>
```

An explicit symbol list creates a custom run. A custom smoke test—even a newer one—never replaces the default canonical candidate result. `scores`, `explain`, and `report` select the latest canonical candidate-universe run unless `--run-id`, `--latest-any`, or a scope filter explicitly requests another saved run. Symbol filters apply to the already selected run and fail clearly when it does not contain a requested symbol.

Use `analysis-list --scope custom`, `analysis-list --date YYYY-MM-DD`, `analysis-list --canonical-only`, and `analysis-list --limit 20` to catalog saved runs. `analysis-show --run-id <id>` is concise; add `--scores`, `--provenance`, or `--full` for detail.

Analysis reports are rendered from exact stored scores and explanations. Their identity includes the as-of date, scope, and short run ID—for example `stock_summary_2026-07-21_candidates_47eed0ae.html`—so same-day candidate and custom reports coexist. Each report has a JSON manifest and a persisted `analysis_reports` record linking hashes and paths to its source run.

Benchmark risk-adjusted metrics are persisted at backtest completion in `backtest_benchmark_metrics`; `benchmark-diagnostics` reads those rows by default. The Phase 3.3 default-universe correction changes CLI orchestration, not the `score_v1` rules or calculations, so strategy version 1.1.0 remains unchanged.

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

- `config/settings.yaml` controls local paths, data source, retry behavior, historical lookback, and log retention (`logs_retention_days`).
- `config/watchlist.csv` defines the static research universe.
- `config/scoring_rules.yaml` defines score weights, thresholds, regime settings, and scoring version.
- `config/backtesting_rules.yaml` defines strategy, portfolio, execution, cost, stop, benchmark, and walk-forward assumptions.
- `config/screening_universe.csv` is the static, additional large-cap list `screen` scans.
- `config/prediction_rules.yaml` configures the experimental `predict` command (horizon, lookback, features, walk-forward folds, regularization).
- `config/trading_rules.yaml` configures `recommend`'s sizing and restrictions (starting capital, position/dollar/trade-count caps); `auto_execute` must stay `false` — no order-placement layer exists yet.

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

SQLite is the system of record. Safe migrations preserve existing prices and add analysis, regime, backtest, trade, fill, equity, metric, walk-forward, experimental-prediction (`prediction_runs`/`prediction_folds`, `gbm_prediction_runs`/`gbm_prediction_folds` shared by predict-v4 and predict-v5 via a `prediction_version` column, with per-fold date ranges, symbol counts, purge counts, baselines, and evaluation provenance), and fundamentals (`fundamentals` — point-in-time SEC EDGAR facts, each row tagged with its own `filed_date`) tables with run identifiers, foreign keys, indexes, uniqueness rules, and transactional writes.

Phase 2 reports contain run metadata, as-of/data-through dates, score version/hash, regime evidence, candidate/risk rankings, components, factors, limitations, quality issues, prior-run changes, methodology, and inline adjusted-price/SMA20/SMA50/SMA200 charts. If a `validate-signals` artifact exists in the reports directory, the report also surfaces the latest Strong Candidate / High Risk bucket's symbol-weighted historical excess return (with its confidence interval and a concentration-warning caveat when applicable) directly above the matching ranking table — dated to that artifact's run, not recomputed per report, and explicitly labeled as a descriptive historical pattern rather than a live prediction for the symbols currently ranked (see "Evaluation honesty" below).

Backtest reports contain assumptions, date/warm-up ranges, universe/exclusions, execution and cost rules, metrics and SPY comparison, inline equity/drawdown charts, period returns, complete trades/rejections, symbol/regime performance, and bias warnings. Separate CSVs cover summary, trades, all signals, rejected candidates, orders/fills, equity, monthly returns, and annual returns. Reports are self-contained and use no CDN.

## Project structure

```text
main.py                         CLI entry point
config/                         Settings, scoring, backtest rules, watchlist
stock_scrapper/analysis/        Indicators-to-score research workflow
stock_scrapper/backtesting/     Configuration, simulation, persistence, metrics, reports
stock_scrapper/collectors/      Daily market-data collection; SEC EDGAR fundamentals (manual/periodic)
stock_scrapper/migrations/      Safe SQLite schema migrations
stock_scrapper/processing/      Validation, indicators, relative strength, point-in-time fundamentals features
stock_scrapper/reporting/       Phase 2 offline reporting and the daily digest
stock_scrapper/portfolio.py     Real-holdings aggregation and hold/sell assessment
stock_scrapper/prediction/      Experimental forward-return prediction (config, dataset, model, service, persistence)
stock_scrapper/trading/         Advisory trade recommendations, sizing, and hindsight review (config, recommendations, review)
scripts/                        Daily automation wrapper, toast notification (Task Scheduler entry point)
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
- **Research scope:** `score_v1` itself is technical-only and omits fundamentals, macroeconomic releases, news, taxes, borrowing constraints, and individual circumstances. `predict-v5` (see below) experimentally adds point-in-time company fundamentals, but only as features for that one experimental model — it changes nothing about `score_v1`'s own scoring or classification logic.
- **Screening universe:** `config/screening_universe.csv` is a hand-maintained illustrative list, not a live feed of any official index's constituents, and is not automatically kept current.
- **Experimental prediction:** `predict` fits a small linear model on freely available technical indicators; its date-grouped, purged, sample-weighted walk-forward holdout accuracy/Brier score are reported next to their own fold-specific baselines precisely so a lack of real edge is visible rather than hidden — and, as of this project's current data, that is exactly what they show (see "Evaluation honesty" above). Past accuracy does not indicate future accuracy. It is not part of `score_v1`.
- **Experimental nonlinear prediction:** `predict-v4` fits a gradient-boosted regression tree ensemble on the same evaluation methodology as `predict`, targeting continuous excess return instead of a binary label; as of this project's current data it is a consistent negative result, losing to its own fold-specific baseline in every fold (see "Evaluation honesty" above). It is not part of `score_v1` or `predict-v3`, and a single extreme prediction (an outlier tree-leaf estimate for a feature combination under-represented in training) should be treated with particular skepticism.
- **Experimental fundamentals-augmented prediction:** `predict-v5` widens `predict-v4` with point-in-time SEC EDGAR fundamentals across all 25 candidates; at a 21-session horizon it is a clean negative result, and at a 252-session (~1 year) horizon, with its own heavier regularization (`predict_v5.gbm` in `config/prediction_rules.yaml`), it is the closest to a real edge anything in this project has shown — the aggregate holdout MSE beats its baseline outright for the first time anywhere in this project, 3 of 5 folds beat their own baseline, and the information coefficient is +0.1910 — but it is still not validated (see "Evaluation honesty" above for the full numbers). Any `predict-v4`/`predict-v5` prediction flagged `[LOW CONFIDENCE]` in the CLI output (more than `OUTLIER_STD_MULTIPLIER`, currently 3, training-target standard deviations from zero) is an extrapolated outlier, not a trustworthy forecast — diagnosed directly for INTC's real +266%–+324% predictions across these runs, traced to a rare, sparsely-precedented feature combination (deeply negative trailing P/E and earnings growth) that heavier regularization alone could not fully tame. `collect-fundamentals`'s CIK resolution and concept-alias coverage had two real bugs (a wrong SEC ticker-to-CIK mapping for XOM; missing noncontrolling-interest equity tag variants) found and fixed this session — neither was a lookahead-safety issue (every fact used is still bounded by its own `filed_date`), but any symbol newly added to the candidate universe should be spot-checked the same way before trusting its fundamentals coverage.
- **Signal validation is descriptive, not inferential:** `validate-signals`' p-values are naive (they assume independent daily trials, which overlapping-horizon, symbol-clustered rows are not) and are labeled as such; the symbol-weighted mean and its confidence interval are the more defensible read, and buckets backed by fewer than four distinct symbols are flagged as concentration-prone rather than presented as population-level evidence.
- **Risk-inversion diagnostics are exploratory:** `investigate-risk-inversion`'s quintile splits and Pearson correlations share `validate-signals`' overlapping-window, symbol-clustered non-independence, and testing several correlated indicators at once can look suggestive by chance. Treat its output as hypothesis generation, not a validated explanation, and it changes nothing about score_v1's scoring, thresholds, or classification logic.
- **Signal-capture diagnostic is exploratory:** `signal-capture-test` is a maximally-permissive score_v1 variant (single-classification entry, no risk-management exits, fixed holding period) built to isolate one question — see "Can the Strong Candidate edge be captured directly?" above. It found removing score_v1's trading rules performs *worse*, not better, suggesting those rules do real protective work rather than suppressing an edge — but it is one exploratory read over one (overlapping-window) history, not a validated conclusion, and changes nothing about score_v1 itself.
- **Advisory recommendations only:** `recommend` sizes suggestions but never places an order — there is no broker integration, and `auto_execute` in `config/trading_rules.yaml` is rejected if set to `true`. Sizing assumes a single account with no existing outside positions, ignores taxes/fees, and derives available cash from `starting_capital` plus recorded lots/sales rather than a real brokerage balance.

## Financial and historical-results disclaimer

Stock Scrapper is educational research software, not a broker, investment adviser, fiduciary, or personalized recommendation service. Nothing produced by the application is an offer or instruction to buy or sell a security.

All scores, classifications, charts, comparisons, and backtests are hypothetical research outputs. Historical or simulated performance does not guarantee future results. Real trading can differ materially because of data revisions, liquidity, spreads, order priority, market impact, taxes, outages, corporate actions, and other factors. You are responsible for independent verification and any decisions you make.
