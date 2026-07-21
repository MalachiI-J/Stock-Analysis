# Phase 3.2 baseline audit — 2026-07-21

Captured before Phase 3.2 code or schema changes.

- Offline tests: **147 passed** in 11.20 seconds.
- Schema/application/strategy: 4 / 0.4.0 / score_v1 1.0.0.
- Price history: 18,825 complete rows, 15 symbols, 1,255 rows each,
  2021-07-20 through 2026-07-20.
- Revisions: 1,603. AAPL has 754 (753 adjusted-close-only); MSFT has
  732 (731 adjusted-close-only). Each other symbol has 9.
- Corporate actions: 59. AAPL and MSFT have 20 each; coverage for several
  other symbols is recent or absent and no explicit checked-range record exists.
- Runs: 7 analysis, 13 backtest, and 3 walk-forward.
- Recent backtests have application/source provenance but null requested/effective
  start, warm-up, universe, and benchmark-sufficiency metadata.
- Recent analyses have scoring version but null software provenance, health, and
  universe metadata.
- Current `data-health` reports Healthy but does not yet test missing exchange
  sessions, action coverage, adjustment anomalies, or revision materiality.

## Schema-v5 migration plan

The migration will be transactional and idempotent. It will preserve every price,
revision, action, and run row while:

1. Extending revision audit rows with class, materiality, absolute/relative
   deltas, and review metadata. Existing rows will be classified, never deleted.
2. Adding one coverage row per symbol/source to record attempted and confirmed
   corporate-action date ranges, including confirmed no-action responses.
3. Completing backtest date/warm-up/provenance columns and adding missing
   requested/effective-end, exclusions, health/action/revision snapshots.
4. Adding diagnostic tables for benchmark metrics, symbol attribution,
   post-signal outcomes, exits, and daily opportunity-cost evidence, all keyed
   to existing backtest runs with cascading foreign keys and idempotent keys.
5. Extending analysis provenance with a deterministic data hash where practical.

No price row will be deleted or rewritten by the migration. A failed migration
will roll back as one transaction. Provider refreshes remain separate explicit
operations.
