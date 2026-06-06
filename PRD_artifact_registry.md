# Product Requirements Document
## Generic Artefact Registry with LLM-Driven Analysis

**Version:** 1.0  
**Status:** Draft  
**Author:** —  
**Date:** June 2026

---

## 1. Purpose

This document specifies a generic, reusable **Artefact Registry** — a runtime store for named tabular datasets that can be populated from external computation engines, user file uploads, or derived analysis outputs. On top of the registry sits an LLM-powered analysis layer that can write and execute sandboxed Python against any registered artefact, with the ability to persist derived outputs back into the registry for chained analysis.

The registry is domain-agnostic. It has no knowledge of how artefacts were produced or what they represent. Domain-specific semantics live in the descriptions attached to each artefact, which the LLM reads to reason about what analyses are meaningful.

---

## 2. Goals

- Provide a single, consistent interface for registering, retrieving, and describing tabular datasets regardless of their origin.
- Enable an LLM to perform open-ended analysis against registered artefacts without being pre-programmed with specific analytical routines.
- Allow derived analysis outputs to be registered back as first-class artefacts, enabling multi-step chained reasoning.
- Accept user-uploaded files as artefacts via a chat interface, with automatic schema inference and LLM-generated descriptions.
- Execute LLM-generated analysis code in a sandboxed environment that prevents filesystem, network, and interpreter escape.
- Provide a test harness with synthetic data, valid and violating code examples, and a local chat UI for end-to-end validation.

---

## 3. Out of Scope

- Persistence of artefacts across sessions (registry is in-memory per session).
- Non-tabular artefacts (images, text blobs, binary files).
- Authentication, authorisation, or multi-user isolation.
- Any specific domain logic (the registry has no concept of finance, biology, logistics, etc.).

---

## 4. Core Concepts

### 4.1 Artefact

A named, described, typed dataset entry in the registry. Each artefact carries:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier within the session registry |
| `data` | `pd.DataFrame \| pd.Series \| scalar` | The actual data object |
| `description` | `str` | Human- and LLM-readable description of content and meaning |
| `dtype_summary` | `str` | Auto-generated shape/type summary (e.g. `DataFrame 1200 × 5`) |
| `provenance` | `enum` | `engine` / `user_upload` / `derived` |
| `parent_artefacts` | `list[str]` | Names of artefacts this was derived from (empty for source artefacts) |
| `created_at` | `datetime` | UTC timestamp of registration |

### 4.2 Registry

A session-scoped container for artefacts, exposing:

- `register(name, data, description, provenance, parents)` — add or overwrite an artefact
- `get(name)` — retrieve the data object by name
- `manifest()` — return a compact string listing all artefacts with their summaries and descriptions, suitable for injection into an LLM system prompt
- `names()` — list of registered names
- `clear()` — remove all artefacts

### 4.3 Manifest

A plain-text summary of the registry contents, injected into the LLM context window at each turn. The manifest gives the LLM full visibility of what data is available without passing the data itself. Example:

```
📦 [engine]      `daily_returns`        Series len=1,200        Daily total return series for the primary dataset
📦 [engine]      `position_history`     DataFrame 1,200 × 847   Daily weight per constituent, columns are security IDs
📤 [user_upload] `external_benchmark`   Series len=1,200        User uploaded: daily returns for a reference series
🔬 [derived]     `sector_te_2022`       DataFrame 250 × 11      Derived from position_history, daily_returns: sector tracking error decomposition, 2022
```

### 4.4 Executor

A sandboxed Python execution environment that:

- Receives a code string and a list of named artefacts to load into the namespace.
- Performs a two-layer safety check: AST-level (pre-execution) and restricted globals (runtime).
- Executes the code in an isolated subprocess with a configurable timeout.
- Requires the code to assign its output to a variable named `result`.
- Returns an `ExecutionResult` containing the output, stdout, and any error.

### 4.5 Ingester

A file parsing and classification component that:

- Accepts raw bytes and a filename from a chat file upload.
- Parses CSV, XLSX, Parquet, and JSON formats.
- Attempts to detect and parse a date/datetime index column.
- Classifies the file's likely role (source data vs reference data vs ambiguous) using heuristics.
- Calls the LLM to generate a one-sentence plain-English description.
- Returns an `IngestedFile` ready for registration.

---

## 5. Functional Requirements

### 5.1 Registry

**FR-REG-01** The registry shall store artefacts keyed by name. Registering a name that already exists overwrites the previous entry without error.

**FR-REG-02** The registry shall auto-generate a `dtype_summary` for any `pd.DataFrame`, `pd.Series`, `np.ndarray`, or scalar value registered.

**FR-REG-03** The registry shall produce a `manifest()` string that includes name, provenance tag, dtype summary, and description for every registered artefact. Derived artefacts shall additionally list their parent names.

**FR-REG-04** The registry shall expose a `get(name)` method that raises `KeyError` with a helpful message listing available names if the requested name is absent.

**FR-REG-05** Provenance values shall be restricted to the enum `{engine, user_upload, derived}`. Any attempt to register with an unknown provenance shall raise `ValueError`.

### 5.2 Executor

**FR-EXE-01** The executor shall reject any code containing `import` or `from ... import` statements at the AST level before execution.

**FR-EXE-02** The executor shall reject any code referencing a forbidden name (`os`, `sys`, `subprocess`, `open`, `exec`, `eval`, `__import__`, `__builtins__`, `__class__`, `__subclasses__`, `importlib`, `pathlib`, `socket`, `shutil`).

**FR-EXE-03** The executor shall reject any code accessing dunder attributes (attributes whose names begin with `__`).

**FR-EXE-04** The executor shall run accepted code in a child process with a configurable timeout (default 30 seconds). Processes exceeding the timeout shall be killed and an error returned.

**FR-EXE-05** The execution namespace shall expose only: `pd` (pandas), `np` (numpy), and the artefact data objects explicitly requested by the caller. No stdlib modules shall be available.

**FR-EXE-06** The executor shall require the executed code to assign a variable named `result`. If no such variable exists after execution, an error shall be returned.

**FR-EXE-07** The executor shall support a `dry_run=True` mode that performs all safety checks but does not execute the code. This mode returns success if the code passes all checks.

**FR-EXE-08** The executor shall capture stdout during execution and include it in the returned `ExecutionResult`.

**FR-EXE-09** The executor shall accept a `requested_artefacts` list. If any named artefact is absent from the registry, execution shall be refused and the error shall list available names.

### 5.3 Ingester

**FR-ING-01** The ingester shall accept raw bytes and a filename. Supported extensions: `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json`. Unsupported extensions shall raise `ValueError` with a clear message.

**FR-ING-02** The ingester shall attempt to identify a date or datetime column by name heuristic (`date`, `time`, `timestamp`, `dt` in column name). If found and parseable, it shall set that column as the DataFrame index sorted ascending.

**FR-ING-03** The ingester shall produce a classification (`source_data`, `reference_data`, `ambiguous`) based on column name heuristics. This classification is advisory; the user may override it in the chat.

**FR-ING-04** The ingester shall call the LLM with the file's schema and first three rows to produce a one-sentence plain-English description. If the LLM call fails, a fallback description of `"Uploaded file: {filename}"` shall be used.

**FR-ING-05** The ingester shall produce a `suggested_artefact_name` by slugifying the filename (lowercase, spaces and special characters replaced with underscores, extension stripped).

### 5.4 Chat Interface

**FR-CHAT-01** The chat interface shall accept a text message and an optional file attachment per turn.

**FR-CHAT-02** When a file is attached, the interface shall display an attachment pill showing the filename before submission.

**FR-CHAT-03** On receipt of a file upload, the backend shall ingest the file, register it in the session registry, and include an ingestion summary in the LLM context before the LLM generates its response.

**FR-CHAT-04** The LLM system prompt shall always include the current registry manifest.

**FR-CHAT-05** When the LLM calls `run_analysis`, the intent summary (plain-English description of what the code does) shall be displayed to the user before execution. The user shall have the option to approve or skip.

**FR-CHAT-06** After successful execution, if the result is a `pd.DataFrame` or `pd.Series`, the chat shall offer to register it as a derived artefact with a name and description provided by the LLM.

**FR-CHAT-07** The chat shall display `ExecutionResult` errors in a styled error block, not as raw tracebacks.

---

## 6. Non-Functional Requirements

**NFR-01 Security:** The executor must not allow code that can read or write to the filesystem, make network calls, or access the parent process's memory. Two-layer enforcement (AST + restricted globals) is mandatory.

**NFR-02 Isolation:** Code execution shall occur in a child process (via `multiprocessing`), not in the main application process. A crashed or OOM child shall not affect the parent.

**NFR-03 Performance:** The executor shall return results for typical analytical operations (DataFrame groupby, rolling window, correlation) within 10 seconds on a standard development machine.

**NFR-04 Extensibility:** The registry's `register` method shall accept any object as `data`. Type-specific `dtype_summary` generation shall be implemented for `pd.DataFrame`, `pd.Series`, and `np.ndarray`; all other types shall fall back to `type(data).__name__`.

**NFR-05 Observability:** All executor calls shall log: artefacts requested, code hash, execution duration, and success/failure. No code content shall be logged at INFO level (only at DEBUG).

---

## 7. Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import pandas as pd
import numpy as np

class Provenance(str, Enum):
    ENGINE      = "engine"
    USER_UPLOAD = "user_upload"
    DERIVED     = "derived"

@dataclass
class Artefact:
    name: str
    data: Any
    description: str
    dtype_summary: str
    provenance: Provenance
    created_at: datetime = field(default_factory=datetime.utcnow)
    parent_artefacts: list[str] = field(default_factory=list)

@dataclass
class ExecutionResult:
    success: bool
    output: Any = None
    stdout: str = ""
    error: str = ""
    code: str = ""
    duration_ms: float = 0.0

@dataclass
class IngestedFile:
    filename: str
    suggested_artefact_name: str
    df: pd.DataFrame
    classification: str          # "source_data" | "reference_data" | "ambiguous"
    description: str
    warnings: list[str]
```

---

## 8. API Surface

### Registry

```python
class ArtefactRegistry:
    def register(
        self,
        name: str,
        data: Any,
        description: str,
        provenance: Provenance = Provenance.DERIVED,
        parents: list[str] = None,
    ) -> Artefact: ...

    def get(self, name: str) -> Any: ...
    def get_artefact(self, name: str) -> Artefact: ...
    def manifest(self) -> str: ...
    def names(self) -> list[str]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
```

### Executor

```python
def run_analysis(
    code: str,
    registry: ArtefactRegistry,
    requested_artefacts: list[str],
    timeout: int = 30,
    dry_run: bool = False,
) -> ExecutionResult: ...
```

### Ingester

```python
def ingest_file(
    filename: str,
    raw_bytes: bytes,
    llm_describe: Callable[[str, pd.DataFrame], str] | None = None,
) -> IngestedFile: ...
```

---

## 9. Testing Requirements

### 9.1 Synthetic Test Data

The test suite shall include a `fixtures.py` module that generates reproducible synthetic datasets without any external dependencies. All fixtures share consistent `instrument_id`, `ISIN`, `SEDOL`, and `CUSIP` identifiers so they can be joined without key engineering.

| Function | Returns | Index | Columns |
|---|---|---|---|
| `make_stock_returns_series()` | `DataFrame` | `date` — 5 years of business days (~1 305 rows) | `instrument_id` — 1 000 stocks; values are daily total return r_t = p_t/p_{t-1}−1 with GARCH-like vol clustering |
| `make_position_history()` | `DataFrame` | `(effective_date, instrument_id)` — MultiIndex; effective dates are monthly rebalancing dates | `benchmark_weight`, `index_weight`, `mcap_usd`; both weight columns sum to 1.0 for every `effective_date` |
| `make_factor_exposures()` | `DataFrame` | `(effective_date, instrument_id)` — MultiIndex | `factor_id` — "Value", "Quality", "Momentum", "Low Vol", "Size"; approximately mean-zero, unit-variance cross-sectionally |
| `make_user_signals()` | `DataFrame` | `(ISIN, signal_date)` — MultiIndex; `signal_date` is weekly (Wednesdays), intentionally misaligned with `effective_date` | "Proprietary 1", "Proprietary 2" |
| `make_universe_file()` | `DataFrame` | `instrument_id` | `stock_name`, `region`, `industry`, `ISIN`, `freefloat`, `SEDOL`, `CUSIP` |

All fixtures shall use a fixed random seed for reproducibility.

### 9.2 Manifest Tests

- Empty registry produces a manifest with a clear "no artefacts registered" message.
- Each provenance type produces the correct icon/tag in the manifest.
- Derived artefacts list their parents in the manifest.
- Manifest is a plain string (no rich formatting) suitable for LLM prompt injection.

### 9.3 Valid Code Examples

The test suite shall include a `valid_code_examples.py` module with at least the following cases, each expected to pass AST check, execute successfully, and return a non-None `result`:

| ID | Description | Expected result type |
|---|---|---|
| `VC-01` | Compute rolling 20-day mean of a Series | `pd.Series` |
| `VC-02` | Compute pairwise correlation of a DataFrame | `pd.DataFrame` |
| `VC-03` | GroupBy + mean aggregation | `pd.DataFrame` |
| `VC-04` | Boolean filter and sum | scalar |
| `VC-05` | Merge two DataFrames on index | `pd.DataFrame` |
| `VC-06` | Cumulative product (return index) | `pd.Series` |
| `VC-07` | Describe / summary statistics | `pd.DataFrame` |
| `VC-08` | np.corrcoef between two series | `np.ndarray` |
| `VC-09` | Multi-step: compute, filter, then aggregate | scalar |
| `VC-10` | Print statement (stdout captured, result still returned) | any |

### 9.4 Violating Code Examples

The test suite shall include a `violating_code_examples.py` module with at least the following cases, each expected to be **rejected before execution** (dry_run catches all):

| ID | Description | Expected failure layer |
|---|---|---|
| `VL-01` | `import os` | AST |
| `VL-02` | `from pathlib import Path` | AST |
| `VL-03` | `import sys; sys.exit()` | AST |
| `VL-04` | `open("/etc/passwd")` | AST (forbidden name) |
| `VL-05` | `eval("1+1")` | AST (forbidden name) |
| `VL-06` | `exec("import socket")` | AST (forbidden name) |
| `VL-07` | `__import__("os")` | AST (forbidden name) |
| `VL-08` | `().__class__.__bases__[0].__subclasses__()` | AST (dunder) |
| `VL-09` | Code with no `result =` assignment | Runtime (missing result) |
| `VL-10` | Infinite loop (timeout) | Timeout kill |
| `VL-11` | Request a non-existent artefact | Pre-execution (missing artefact) |

### 9.5 Integration Tests

- Register two artefacts, run code that joins them, register the output as derived, verify manifest shows parent linkage.
- Upload a synthetic CSV file, verify ingestion produces correct schema, description, and artefact registration.
- Run a sequence of three chained analyses, each consuming the previous step's output.

### 9.6 Test Chat UI

A self-contained local chat application (single HTML file or lightweight Python server) for manual end-to-end validation, specified in Section 10.

---

## 10. Test Chat UI

### 10.1 Purpose

A local, dependency-light chat interface for interactive testing of the full stack: file upload → ingestion → manifest display → LLM analysis → sandboxed execution → result display → derived artefact registration.

### 10.2 Technology

Single-page React application served from a Python FastAPI backend. No build step required — React loaded via CDN. The UI connects to the Anthropic API via the backend (API key in environment variable, never sent to the browser).

### 10.3 Layout

```
┌─────────────────────────────────────────────────────────┐
│  ARTEFACT REGISTRY CHAT                                  │
├───────────────────────┬─────────────────────────────────┤
│                       │                                  │
│   REGISTRY MANIFEST   │   CONVERSATION                  │
│   (live-updating)     │                                  │
│                       │   [user message]                 │
│  📦 returns           │   [assistant response]           │
│  📦 positions         │   [intent approval card]         │
│  📤 benchmark         │   [result table/chart]           │
│  🔬 derived_1         │                                  │
│                       │                                  │
│  [Clear Registry]     │   ┌────────────────────────┐    │
│  [Load Test Data]     │   │ 📎 file.csv  [×]       │    │
│                       │   │ [message input........] │    │
│                       │   │              [Send  →]  │    │
│                       │   └────────────────────────┘    │
└───────────────────────┴─────────────────────────────────┘
```

### 10.4 UI Behaviours

**Manifest panel:** Refreshes after every turn. Each artefact row is colour-coded by provenance (engine=blue, user_upload=amber, derived=green). Clicking a row shows the first 5 rows of the data in a tooltip.

**Intent approval card:** When the LLM proposes to run code, display a card showing the plain-English intent description with [Run] and [Skip] buttons. Code is hidden by default with a [Show code] toggle for technical users.

**Result rendering:** DataFrames render as scrollable tables (max 10 rows shown, "show more" toggle). Series render as a simple two-column table. Scalars render inline in the assistant message. Errors render in a red-bordered block with the error message only (no raw traceback).

**Register derived artefact:** After a successful execution returning a DataFrame or Series, show a small form: name field (pre-filled with LLM suggestion) and description field. [Register] button calls the backend to add to the registry.

**Load Test Data button:** Calls `/api/load-test-data` which registers all synthetic fixtures into the registry and returns the updated manifest. Allows immediate testing without a file upload.

### 10.5 Backend Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Main chat turn. Accepts `multipart/form-data`: `message`, optional `file`, `session_id`. Returns streaming SSE. |
| `GET` | `/api/manifest` | Returns current session manifest as plain text. |
| `POST` | `/api/register-derived` | Body: `{name, description, execution_id}`. Registers the output of a prior execution as a derived artefact. |
| `POST` | `/api/load-test-data` | Registers all synthetic test fixtures. Returns manifest. |
| `DELETE` | `/api/registry` | Clears the session registry. |

---

## 11. Acceptance Criteria

| # | Criterion |
|---|---|
| AC-01 | All 10 valid code examples execute successfully and return the expected result type |
| AC-02 | All 11 violating code examples are blocked before execution with a clear error message |
| AC-03 | Uploading a CSV file in the test chat UI results in a registered artefact visible in the manifest within one turn |
| AC-04 | A chain of three analyses, each consuming the previous output, completes successfully with parent linkage visible in the manifest |
| AC-05 | A timed-out analysis (infinite loop) is killed within `timeout + 2` seconds and returns an error without crashing the server |
| AC-06 | The registry manifest, when injected into the LLM prompt, enables the LLM to correctly identify joinable artefacts by index without being explicitly told |
| AC-07 | All tests in the test suite pass with `pytest` against a clean environment |

---

## 12. Open Questions

| # | Question | Owner |
|---|---|---|
| OQ-01 | Should the registry support non-tabular artefacts (dicts, scalars registered explicitly)? Scalars currently fall back to `type.__name__` summary. | — |
| OQ-02 | For large DataFrames (>10M cells), should `get()` return a lazy proxy rather than the full object to avoid subprocess serialisation overhead? | — |
| OQ-03 | Should derived artefacts be immutable once registered, or allow overwrite? Current spec allows overwrite. | — |
| OQ-04 | Docker-based executor isolation: when is the subprocess approach insufficient and a container boundary required? | — |
| OQ-05 | Should the manifest include a `sample` (first 2 rows) for each artefact to help the LLM understand data shape without running code? | — |
