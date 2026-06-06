"""
test_artifact_registry.py
--------------------------
pytest suite covering: registry, manifest, executor (valid + violating), ingester, chained analysis.

Run with:
    pip install pytest pandas numpy openpyxl pyarrow
    pytest test_artifact_registry.py -v
"""

from __future__ import annotations

import io
import pytest
import numpy as np
import pandas as pd

from artifact_registry import (
    ArtefactRegistry,
    ExecutionResult,
    IngestedFile,
    Provenance,
    _ast_check,
    ingest_file,
    run_analysis,
)
from fixtures import (
    make_all_fixtures,
    make_returns_csv_bytes,
    make_universe_csv_bytes,
)
from code_examples import VALID_EXAMPLES, VIOLATING_EXAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def registry() -> ArtefactRegistry:
    reg = ArtefactRegistry()
    descriptions = {
        "universe":         "Static universe file: 1 000 stocks with region, industry, ISIN, SEDOL, CUSIP",
        "stock_returns":    "Daily total return matrix: 5 years of business days × 1 000 instruments",
        "position_history": "Monthly rebalancing weights (benchmark + index) and market cap per stock",
        "factor_exposures": "Factor loading matrix at monthly rebalancing dates: Value, Quality, Momentum, Low Vol, Size",
        "user_signals":     "Weekly proprietary signals per ISIN: Proprietary 1, Proprietary 2",
    }
    for name, data in make_all_fixtures().items():
        reg.register(
            name=name,
            data=data,
            description=descriptions[name],
            provenance=Provenance.ENGINE,
        )
    return reg


@pytest.fixture
def empty_registry() -> ArtefactRegistry:
    return ArtefactRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_register_and_retrieve(self, registry):
        data = registry.get("stock_returns")
        assert isinstance(data, pd.DataFrame)
        assert data.shape[1] == 1000

    def test_overwrite(self, registry):
        new_series = pd.Series([1.0, 2.0, 3.0])
        registry.register("stock_returns", new_series, "overwritten", Provenance.DERIVED)
        assert len(registry.get("stock_returns")) == 3

    def test_get_missing_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="not found"):
            registry.get("does_not_exist")

    def test_key_error_lists_available_names(self, registry):
        with pytest.raises(KeyError) as exc_info:
            registry.get("missing")
        assert "stock_returns" in str(exc_info.value)

    def test_invalid_provenance_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown provenance"):
            registry.register("x", pd.Series([1]), "desc", provenance="bad_value")

    def test_names(self, registry):
        names = registry.names()
        assert "stock_returns" in names
        assert "position_history" in names
        assert "universe" in names

    def test_len(self, registry):
        assert len(registry) == 5

    def test_clear(self, registry):
        registry.clear()
        assert len(registry) == 0

    def test_dtype_summary_position_history(self, registry):
        art = registry.get_artefact("position_history")
        assert "60,000" in art.dtype_summary
        assert "3" in art.dtype_summary

    def test_dtype_summary_stock_returns(self, registry):
        art = registry.get_artefact("stock_returns")
        assert "DataFrame" in art.dtype_summary
        assert "1,000" in art.dtype_summary

    def test_parent_artefacts_recorded(self, registry):
        registry.register(
            "derived_thing",
            pd.Series([1.0]),
            "a derived artefact",
            Provenance.DERIVED,
            parents=["stock_returns", "factor_exposures"],
        )
        art = registry.get_artefact("derived_thing")
        assert "stock_returns" in art.parent_artefacts
        assert "factor_exposures" in art.parent_artefacts


# ─────────────────────────────────────────────────────────────────────────────
# Manifest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestManifest:

    def test_empty_manifest(self, empty_registry):
        assert "no artefacts" in empty_registry.manifest()

    def test_manifest_contains_all_names(self, registry):
        m = registry.manifest()
        for name in registry.names():
            assert name in m

    def test_manifest_engine_icon(self, registry):
        assert "📦" in registry.manifest()

    def test_manifest_user_upload_icon(self, registry):
        registry.register("uploaded", pd.Series([1.0]), "a upload", Provenance.USER_UPLOAD)
        assert "📤" in registry.manifest()

    def test_manifest_derived_icon(self, registry):
        registry.register("derived", pd.Series([1.0]), "a derived", Provenance.DERIVED)
        assert "🔬" in registry.manifest()

    def test_manifest_shows_parents(self, registry):
        registry.register(
            "child",
            pd.Series([1.0]),
            "child artefact",
            Provenance.DERIVED,
            parents=["stock_returns"],
        )
        m = registry.manifest()
        assert "derived from" in m
        assert "stock_returns" in m

    def test_manifest_is_plain_string(self, registry):
        m = registry.manifest()
        assert isinstance(m, str)

    def test_manifest_no_data_in_output(self, registry):
        """Manifest must not contain raw data values."""
        m = registry.manifest()
        # position_history benchmark_weight values like 0.00123 should not appear
        assert "0.00123" not in m


# ─────────────────────────────────────────────────────────────────────────────
# AST check unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAstCheck:

    def test_clean_code_passes(self):
        assert _ast_check("result = 1 + 1") == []

    def test_import_blocked(self):
        violations = _ast_check("import os")
        assert any("import" in v for v in violations)

    def test_from_import_blocked(self):
        violations = _ast_check("from pathlib import Path")
        assert any("import" in v for v in violations)

    def test_forbidden_name_blocked(self):
        violations = _ast_check("open('/etc/passwd')")
        assert any("open" in v for v in violations)

    def test_dunder_attr_blocked(self):
        violations = _ast_check("x.__class__.__bases__")
        assert any("dunder" in v for v in violations)

    def test_syntax_error_returned(self):
        violations = _ast_check("def (broken:")
        assert any("SyntaxError" in v for v in violations)

    def test_deduplicated_violations(self):
        code = "import os\nimport sys"
        violations = _ast_check(code)
        assert len([v for v in violations if "import" in v]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Valid code execution tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidExecution:

    @pytest.mark.parametrize("example", VALID_EXAMPLES, ids=[e["id"] for e in VALID_EXAMPLES])
    def test_valid_example(self, registry, example):
        result = run_analysis(
            code=example["code"],
            registry=registry,
            requested_artefacts=example["artefacts"],
        )
        assert result.success, f"{example['id']} failed: {result.error}"
        assert result.output is not None, f"{example['id']} returned None result"
        assert isinstance(result.output, example["expected_result_type"]), (
            f"{example['id']}: expected {example['expected_result_type']}, "
            f"got {type(result.output)}"
        )

    def test_stdout_captured(self, registry):
        code = 'print("hello from sandbox")\nresult = 42'
        res = run_analysis(code, registry, ["stock_returns"])
        assert res.success
        assert "hello from sandbox" in res.stdout

    def test_duration_recorded(self, registry):
        code = "result = stock_returns.iloc[:, 0].mean()"
        res = run_analysis(code, registry, ["stock_returns"])
        assert res.duration_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# Violating code tests
# ─────────────────────────────────────────────────────────────────────────────

class TestViolatingExecution:

    @pytest.mark.parametrize(
        "example",
        [e for e in VIOLATING_EXAMPLES if e["expected_failure_layer"] != "timeout"],
        ids=[e["id"] for e in VIOLATING_EXAMPLES if e["expected_failure_layer"] != "timeout"],
    )
    def test_violating_example(self, registry, example):
        result = run_analysis(
            code=example["code"],
            registry=registry,
            requested_artefacts=example["artefacts"],
        )
        assert not result.success, f"{example['id']} should have been blocked"
        assert example["expected_error_contains"].lower() in result.error.lower(), (
            f"{example['id']}: expected '{example['expected_error_contains']}' "
            f"in error, got: {result.error}"
        )

    def test_timeout_kills_process(self, registry):
        example = next(e for e in VIOLATING_EXAMPLES if e["id"] == "VL-10")
        result = run_analysis(
            code=example["code"],
            registry=registry,
            requested_artefacts=example["artefacts"],
            timeout=example["timeout_override"],
        )
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_dry_run_catches_violations(self, registry):
        """Dry run should catch all AST violations without executing."""
        code = "import os\nresult = os.listdir('.')"
        result = run_analysis(code, registry, ["stock_returns"], dry_run=True)
        assert not result.success
        assert "import" in result.error.lower()

    def test_dry_run_passes_valid_code(self, registry):
        code = "result = stock_returns.iloc[:, 0].mean()"
        result = run_analysis(code, registry, ["stock_returns"], dry_run=True)
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# Ingester tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngester:

    def test_csv_ingestion(self):
        raw = make_returns_csv_bytes()
        result = ingest_file("returns_2024.csv", raw)
        assert isinstance(result.df, pd.DataFrame)
        assert isinstance(result.df.index, pd.DatetimeIndex)

    def test_universe_csv_classification(self):
        raw = make_universe_csv_bytes()
        result = ingest_file("universe.csv", raw)
        assert result.classification == "reference_data"

    def test_returns_csv_classification(self):
        # Column names are instrument IDs (INST_xxxx) — no source/reference hint
        raw = make_returns_csv_bytes()
        result = ingest_file("returns.csv", raw)
        assert result.classification == "ambiguous"

    def test_suggested_name_slugified(self):
        raw = make_universe_csv_bytes()
        result = ingest_file("My Universe File 2024.csv", raw)
        assert " " not in result.suggested_artefact_name
        assert result.suggested_artefact_name == "my_universe_file_2024"

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ingest_file("data.psd", b"not a real file")

    def test_fallback_description_without_llm(self):
        raw = make_returns_csv_bytes()
        result = ingest_file("returns.csv", raw)
        assert "returns.csv" in result.description

    def test_custom_llm_describe(self):
        raw = make_returns_csv_bytes()
        result = ingest_file(
            "returns.csv",
            raw,
            llm_describe=lambda fn, df: "A daily return matrix from a unit test",
        )
        assert result.description == "A daily return matrix from a unit test"

    def test_failed_llm_describe_falls_back(self):
        raw = make_returns_csv_bytes()
        def bad_llm(fn, df):
            raise RuntimeError("LLM unavailable")
        result = ingest_file("returns.csv", raw, llm_describe=bad_llm)
        assert "returns.csv" in result.description
        assert len(result.warnings) > 0

    def test_parquet_round_trip(self):
        df = make_all_fixtures()["position_history"]
        buf = io.BytesIO()
        df.to_parquet(buf)
        result = ingest_file("positions.parquet", buf.getvalue())
        assert result.df.shape == df.shape


# ─────────────────────────────────────────────────────────────────────────────
# Integration: chained analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestChainedAnalysis:

    def test_three_step_chain(self, registry):
        """
        Step 1: cumulative return for first stock → register as derived
        Step 2: drawdown series from cum_returns → register as derived
        Step 3: date of worst drawdown → scalar string
        Each step depends on the previous.
        """
        # Step 1
        r1 = run_analysis(
            "result = (1 + stock_returns.iloc[:, 0]).cumprod()",
            registry, ["stock_returns"]
        )
        assert r1.success
        registry.register("cum_returns", r1.output, "Cumulative return index",
                          Provenance.DERIVED, parents=["stock_returns"])

        # Step 2
        r2 = run_analysis(
            """
rolling_max = cum_returns.expanding().max()
result = (cum_returns - rolling_max) / rolling_max
""",
            registry, ["cum_returns"]
        )
        assert r2.success
        registry.register("drawdowns", r2.output, "Drawdown series",
                          Provenance.DERIVED, parents=["cum_returns"])

        # Step 3
        r3 = run_analysis(
            "result = str(drawdowns.idxmin().date())",
            registry, ["drawdowns"]
        )
        assert r3.success
        assert isinstance(r3.output, str)

        # Verify manifest shows the full lineage
        m = registry.manifest()
        assert "cum_returns" in m
        assert "drawdowns" in m
        assert "derived from" in m

    def test_join_uploaded_with_engine_artefact(self, registry):
        """User uploads universe CSV; code joins latest position weights with universe metadata."""
        raw = make_universe_csv_bytes()
        ingested = ingest_file("universe_upload.csv", raw)
        registry.register(
            "uploaded_universe",
            ingested.df,
            ingested.description,
            Provenance.USER_UPLOAD,
        )

        code = """
latest_date = position_history.index.get_level_values("effective_date").max()
pos = position_history.loc[latest_date]
result = pos.join(uploaded_universe[["region", "industry"]])
"""
        r = run_analysis(code, registry, ["position_history", "uploaded_universe"])
        assert r.success
        assert isinstance(r.output, pd.DataFrame)
        # 3 weight/mcap columns + 2 metadata columns
        assert r.output.shape[1] == 5

    def test_derived_artefact_in_manifest_shows_parents(self, registry):
        r = run_analysis(
            "result = stock_returns.rolling(20).std()",
            registry, ["stock_returns"]
        )
        assert r.success
        registry.register("rolling_vol", r.output, "20-day rolling volatility",
                          Provenance.DERIVED, parents=["stock_returns"])
        m = registry.manifest()
        assert "rolling_vol" in m
        assert "stock_returns" in m
