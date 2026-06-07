# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                        # install all dependencies
uv run pytest -v                               # run all tests
uv run pytest test_artifact_registry.py::test_name -v  # run a single test
uv run python chat_server.py                   # start the chat UI at http://localhost:8000
```

Environment variables for the server:
- `OPENAI_API_KEY` — required
- `OPENAI_MODEL` — override the model (default: `gpt-4o-mini`)

## Architecture

The project is a single-session, in-memory LLM-driven data analysis tool. There are three layers:

### Core library (`artifact_registry.py`)
No FastAPI, no LLM calls — pure pandas/numpy. Three public entry points:
- `ArtefactRegistry` — dict-backed store of `Artefact` dataclasses, each holding the raw data object plus provenance metadata (`engine` | `user_upload` | `derived`). `manifest()` renders a human-readable summary that is injected into every LLM system prompt.
- `run_analysis(code, registry, requested_artefacts)` — executes LLM-generated code safely. Two-stage safety model: (1) AST pre-check blocks imports, forbidden names (`os`, `sys`, `eval`, etc.), and dunder attribute access; (2) execution runs in a forked child process (`mp.get_context("fork")`) with stripped `__builtins__` and only `pd`/`np` available. Code **must** assign its output to a variable named `result`. The parent drains the queue before joining to avoid pipe-buffer deadlock with large DataFrames.
- `ingest_file(filename, raw_bytes, llm_describe?)` — parses CSV/XLSX/Parquet/JSON, auto-detects date columns, classifies the file as `source_data` / `reference_data` / `ambiguous` by column-name heuristics, and optionally calls an LLM callback for a one-sentence description.

### Server (`chat_server.py`)
FastAPI app with an in-memory `Session` store (registry + chat history + pending execution results). Key flow:

1. `POST /api/chat` — appends to session history, calls `_llm_chat()` with the `run_analysis` tool, stores the pending code under a UUID in `session.pending_results`, and returns the UUID to the client without running anything yet.
2. `POST /api/execute` — looks up the pending UUID and actually calls `run_analysis()`. Kept separate so the user can inspect/skip code before it runs.
3. `POST /api/register-derived` — promotes the output of a successful execution into the registry as a `DERIVED` artefact, with the source artefact names recorded as parents.

The special `/execute <code>` chat command bypasses the LLM and runs code directly.

### Frontend (`chat_server.py` — `REACT_APP` string)
A React 18 SPA (Babel transpiled in-browser, no build step) served inline from the `/` route. All state is local to the browser tab; the `SESSION_ID` is a random token generated at page load.

### Test fixtures (`fixtures.py`)
Five synthetic datasets with a shared key space (`instrument_id`, `ISIN`, `SEDOL`, `CUSIP`) for join testing. Fixed random seeds — outputs are deterministic.
