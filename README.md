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

## Test UI walkthrough

1. Open `http://localhost:8000`
2. Click **Load Test Data** — registers 4 synthetic artefacts
3. Ask: *"What does the return distribution look like?"*
4. AIDA proposes code → click **Run** (or **Show code** first)
5. Result renders as table; click **Register** to persist as derived artefact
6. Upload a CSV via 📎 — it appears in the manifest as a `user_upload`
7. Ask: *"Compare my upload against the returns"* — AIDA joins them

## Safety layers

| Layer | What it catches |
|---|---|
| AST pre-check | `import`, `from ... import`, forbidden names (`os`, `eval`, ...), dunder attributes |
| Restricted globals | No stdlib available at runtime even if AST check missed something |
| Subprocess isolation | Child process crash/OOM doesn't affect server |
| Timeout | Hung/infinite processes killed after configurable seconds |
