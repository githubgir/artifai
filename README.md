# Artefact Registry

Generic, LLM-driven tabular data registry with sandboxed Python execution.

## Files

| File | Purpose |
|---|---|
| `artifact_registry.py` | Core: `ArtefactRegistry`, `run_analysis`, `ingest_file` |
| `fixtures.py` | Synthetic test data generators (fixed seed, no external deps) |
| `code_examples.py` | Catalogue of valid and violating code strings |
| `test_artifact_registry.py` | Full pytest suite |
| `chat_server.py` | FastAPI + React test UI |

## Quick Start

```bash
pip install fastapi uvicorn anthropic pandas numpy openpyxl pyarrow python-multipart pytest

# Run tests
pytest test_artifact_registry.py -v

# Start chat UI
export ANTHROPIC_API_KEY=sk-...
python chat_server.py
# → http://localhost:8000
```

## Synthetic fixtures

`fixtures.py` ships five datasets that share consistent `instrument_id`, `ISIN`, `SEDOL`, and `CUSIP` keys so they can be joined without key engineering:

| Fixture | Shape | Index | Description |
|---|---|---|---|
| `make_universe_file()` | 1 000 × 7 | `instrument_id` | Stock descriptors: name, region, industry, ISIN, freefloat, SEDOL, CUSIP |
| `make_stock_returns_series()` | ~1 305 × 1 000 | `date` (business days, 5 years) | Daily total return matrix with GARCH-like vol clustering |
| `make_position_history()` | ~61 000 × 3 | `(effective_date, instrument_id)` | benchmark_weight, index_weight, mcap_usd; weights sum to 1.0 per date |
| `make_factor_exposures()` | ~61 000 × 5 | `(effective_date, instrument_id)` | Factor loadings: Value, Quality, Momentum, Low Vol, Size |
| `make_user_signals()` | ~260 000 × 2 | `(ISIN, signal_date)` | Weekly proprietary signals (misaligned with monthly rebalance dates) |

## Test UI walkthrough

1. Open `http://localhost:8000`
2. Click **Load Test Data** — registers 5 synthetic artefacts
3. Ask: *"What does the return distribution look like?"*
4. AIDA proposes code → click **Run** (or **Show code** first)
5. Result renders as table; click **Register** to persist as derived artefact
6. Upload a CSV via 📎 — it appears in the manifest as a `user_upload`
7. Ask: *"Join user signals to factor exposures via ISIN"* — AIDA handles the asof join

## Safety layers

| Layer | What it catches |
|---|---|
| AST pre-check | `import`, `from ... import`, forbidden names (`os`, `eval`, ...), dunder attributes |
| Restricted globals | No stdlib available at runtime even if AST check missed something |
| Subprocess isolation | Child process crash/OOM doesn't affect server |
| Timeout | Hung/infinite processes killed after configurable seconds |
