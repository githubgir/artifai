"""
artifact_registry.py
--------------------
Core implementation: ArtefactRegistry, Executor, Ingester.
No domain logic. No external dependencies beyond pandas, numpy.
"""

from __future__ import annotations

import ast
import io
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class Provenance(str, Enum):
    ENGINE      = "engine"
    USER_UPLOAD = "user_upload"
    DERIVED     = "derived"

_PROVENANCE_ICON = {
    Provenance.ENGINE:      "📦",
    Provenance.USER_UPLOAD: "📤",
    Provenance.DERIVED:     "🔬",
}


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


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class ArtefactRegistry:
    def __init__(self):
        self._store: dict[str, Artefact] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        data: Any,
        description: str,
        provenance: Provenance = Provenance.DERIVED,
        parents: list[str] | None = None,
    ) -> Artefact:
        if not isinstance(provenance, Provenance):
            raise ValueError(
                f"Unknown provenance '{provenance}'. "
                f"Valid values: {[p.value for p in Provenance]}"
            )
        art = Artefact(
            name=name,
            data=data,
            description=description,
            dtype_summary=_dtype_summary(data),
            provenance=provenance,
            parent_artefacts=parents or [],
        )
        self._store[name] = art
        return art

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        if name not in self._store:
            raise KeyError(
                f"Artefact '{name}' not found. "
                f"Available: {self.names() or ['(none)']}"
            )
        return self._store[name].data

    def get_artefact(self, name: str) -> Artefact:
        if name not in self._store:
            raise KeyError(
                f"Artefact '{name}' not found. "
                f"Available: {self.names() or ['(none)']}"
            )
        return self._store[name]

    # ── Manifest ──────────────────────────────────────────────────────────────

    def manifest(self) -> str:
        if not self._store:
            return "(no artefacts registered)"
        lines = [
            "SHARED DIMENSIONS: instrument_id, ISIN, SEDOL, CUSIP are consistent join keys across all artefacts.",
            "FREQUENCY NOTE: 'date' is daily (business days); 'effective_date' is monthly (first business day of each month).",
            "  → Always align these two time axes via asof-join or forward-fill before concatenating daily and monthly artefacts.",
            "",
        ]
        for art in self._store.values():
            icon = _PROVENANCE_ICON[art.provenance]
            tag  = f"[{art.provenance.value}]"
            line = f"##{art.name}\n{icon} {tag:<14} `{art.name}`  {art.dtype_summary:<30}  {art.description}"
            if art.parent_artefacts:
                line += f"\n   derived from: {', '.join(art.parent_artefacts)}"
            lines.append(line)
        return "\n".join(lines)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _dtype_summary(data: Any) -> str:
    if isinstance(data, pd.DataFrame):
        return f"DataFrame {data.shape[0]:,} × {data.shape[1]:,}"
    if isinstance(data, pd.Series):
        return f"Series len={len(data):,} dtype={data.dtype}"
    if isinstance(data, np.ndarray):
        return f"ndarray shape={data.shape}"
    return type(data).__name__


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "open", "exec", "eval",
    "__import__", "__builtins__", "__class__", "__subclasses__",
    "importlib", "pathlib", "socket", "shutil",
}

SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "sum": sum, "min": min, "max": max, "abs": abs,
    "round": round, "sorted": sorted, "reversed": reversed,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "print": print, "isinstance": isinstance, "type": type,
    "hasattr": hasattr, "getattr": getattr,
}


def _ast_check(code: str) -> list[str]:
    """Return list of violation strings. Empty list = clean."""
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("import statements are not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"forbidden name: `{node.id}`")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"dunder attribute access not allowed: `.{node.attr}`")

    return list(dict.fromkeys(violations))  # deduplicate, preserve order


def _worker(
    code: str,
    artefact_data: dict,
    result_queue: mp.Queue,
) -> None:
    """Runs in isolated child process."""
    import io as _io
    from contextlib import redirect_stdout

    try:
        import pandas as _pd
        import numpy as _np

        exec_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": _pd,
            "np": _np,
        }
        local_ns = dict(artefact_data)

        stdout_buf = _io.StringIO()
        with redirect_stdout(stdout_buf):
            exec(code, exec_globals, local_ns)  # noqa: S102

        if "result" not in local_ns:
            result_queue.put(("missing_result", stdout_buf.getvalue()))
        else:
            result_queue.put(("ok", local_ns["result"], stdout_buf.getvalue()))

    except Exception:
        result_queue.put(("error", traceback.format_exc()))


def run_analysis(
    code: str,
    registry: ArtefactRegistry,
    requested_artefacts: list[str],
    timeout: int = 30,
    dry_run: bool = False,
) -> ExecutionResult:
    """
    Safely execute LLM-generated code against named registry artefacts.

    Convention: code must assign its final output to `result`.
    """
    # Strip redundant pd/np imports already provided by the execution environment
    import re as _re
    code = _re.sub(r"^\s*import\s+(?:pandas\s+as\s+pd|numpy\s+as\s+np)\s*\n?", "", code, flags=_re.MULTILINE)

    # 1. AST safety check
    violations = _ast_check(code)
    if violations:
        return ExecutionResult(
            success=False,
            error="Code blocked by safety check:\n" + "\n".join(f"  • {v}" for v in violations),
            code=code,
        )

    if dry_run:
        return ExecutionResult(success=True, code=code)

    # 2. Resolve requested artefacts
    artefact_data = {}
    missing = []
    for name in requested_artefacts:
        try:
            artefact_data[name] = registry.get(name)
        except KeyError:
            missing.append(name)
    if missing:
        return ExecutionResult(
            success=False,
            error=(
                f"Artefacts not found: {missing}.\n"
                f"Available: {registry.names() or ['(none)']}"
            ),
            code=code,
        )

    # 3. Execute in child process
    # fork avoids re-importing heavy deps on every call; the child inherits the
    # parent's memory so startup is near-instant.
    ctx = mp.get_context("fork")
    q   = ctx.Queue()
    p   = ctx.Process(target=_worker, args=(code, artefact_data, q))
    t0  = time.monotonic()
    p.start()

    # Read result BEFORE joining. With large DataFrames the child blocks on
    # q.put() because the pipe buffer fills up, and the parent blocks on
    # p.join() — a deadlock. Draining first resolves it.
    try:
        payload = q.get(timeout=timeout)
    except Exception:
        payload = None

    p.join(2)  # give child time to exit cleanly after we have its result
    elapsed_ms = (time.monotonic() - t0) * 1000

    if p.is_alive():
        p.kill()
        p.join()
        return ExecutionResult(
            success=False,
            error=f"Execution timed out after {timeout}s",
            code=code,
            duration_ms=elapsed_ms,
        )

    if payload is None:
        return ExecutionResult(
            success=False,
            error="Child process exited without returning a result",
            code=code,
            duration_ms=elapsed_ms,
        )

    if payload[0] == "ok":
        _, output, stdout = payload
        return ExecutionResult(
            success=True,
            output=output,
            stdout=stdout,
            code=code,
            duration_ms=elapsed_ms,
        )
    elif payload[0] == "missing_result":
        _, stdout = payload
        return ExecutionResult(
            success=False,
            error="Code did not assign a `result` variable.",
            stdout=stdout,
            code=code,
            duration_ms=elapsed_ms,
        )
    else:
        _, tb = payload
        return ExecutionResult(
            success=False,
            error=tb,
            code=code,
            duration_ms=elapsed_ms,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ingester
# ─────────────────────────────────────────────────────────────────────────────

_PARSERS = {
    "csv":     lambda b: pd.read_csv(io.BytesIO(b)),
    "xlsx":    lambda b: pd.read_excel(io.BytesIO(b)),
    "xls":     lambda b: pd.read_excel(io.BytesIO(b)),
    "parquet": lambda b: pd.read_parquet(io.BytesIO(b)),
    "json":    lambda b: pd.read_json(io.BytesIO(b)),
}

_DATE_HINTS = {"date", "time", "timestamp", "dt", "period"}

_SOURCE_HINTS    = {"return", "price", "close", "nav", "value", "level"}
_REFERENCE_HINTS = {"weight", "sedol", "isin", "ticker", "constituent", "id", "code"}


def ingest_file(
    filename: str,
    raw_bytes: bytes,
    llm_describe: Callable[[str, pd.DataFrame], str] | None = None,
) -> IngestedFile:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in _PARSERS:
        raise ValueError(
            f"Unsupported file extension '.{ext}'. "
            f"Supported: {list(_PARSERS)}"
        )

    df = _PARSERS[ext](raw_bytes)
    warnings: list[str] = []

    # Attempt date index detection
    for col in list(df.columns):
        if any(hint in col.lower() for hint in _DATE_HINTS):
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break
            except Exception:
                warnings.append(f"Could not parse column `{col}` as dates")

    # Classification heuristic
    cols_lower = {c.lower() for c in df.columns}
    if cols_lower & _SOURCE_HINTS:
        classification = "source_data"
    elif cols_lower & _REFERENCE_HINTS:
        classification = "reference_data"
    else:
        classification = "ambiguous"

    # Description
    if llm_describe is not None:
        try:
            description = llm_describe(filename, df)
        except Exception as e:
            warnings.append(f"LLM description failed: {e}")
            description = f"Uploaded file: {filename}"
    else:
        description = f"Uploaded file: {filename}"

    # Slug name
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", filename.lower().rsplit(".", 1)[0]).strip("_")

    return IngestedFile(
        filename=filename,
        suggested_artefact_name=slug,
        df=df,
        classification=classification,
        description=description,
        warnings=warnings,
    )
