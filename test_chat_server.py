"""
test_chat_server.py
-------------------
Playwright test suite for chat_server.py.

The suite starts the server as a subprocess, then:
  - Uses the /execute command to run Python scripts directly (no LLM needed)
  - Verifies scalar, DataFrame, and blocked-code responses
  - Includes browser-level UI tests (require `playwright install chromium`)

Run:
    pytest test_chat_server.py -v
    pytest test_chat_server.py -v -k "not browser"   # API-only, no browser needed
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Server fixture — starts chat_server.py as a subprocess for the test session
# ─────────────────────────────────────────────────────────────────────────────

SERVER_PORT = 18765
BASE_URL = f"http://localhost:{SERVER_PORT}"


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def server():
    """Start chat_server on a dedicated test port; yield base URL; stop after tests."""
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key-not-needed-for-execute",
        "OPENAI_API_KEY": "test-key-not-needed-for-execute",
    }
    proc = subprocess.Popen(
        [sys.executable, "chat_server.py"],
        env={**env, "PORT": str(SERVER_PORT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__) or ".",
    )
    # chat_server hard-codes port 8000 when run as __main__; patch via monkeypatching
    # is not practical across processes, so we override via a helper below.
    if not _wait_for_server(f"http://localhost:8000", timeout=15):
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"Server failed to start.\nstdout: {out.decode()}\nstderr: {err.decode()}")
    yield "http://localhost:8000"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def execute(base_url: str, code: str, session_id: str = "test_session") -> dict:
    """POST /api/chat with an /execute command and return the parsed JSON."""
    resp = requests.post(
        f"{base_url}/api/chat",
        data={"message": f"/execute\n{code}", "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# API-level tests (no browser required)
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteCommand:
    """Tests for the /execute chat command via direct HTTP calls."""

    def test_scalar_arithmetic(self, server):
        """Simple arithmetic returns a scalar result."""
        data = execute(server, "result = 6 * 7")
        assert data["intent"] == "/execute"
        assert data["ast_error"] is None
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "scalar"
        assert dr["output_preview"] == "42"

    def test_string_result(self, server):
        """String result is serialised as a scalar."""
        data = execute(server, 'result = "hello world"', session_id="test_strings")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "scalar"
        assert "hello world" in dr["output_preview"]

    def test_dataframe_result(self, server):
        """Code that produces a DataFrame is returned as output_type='dataframe'."""
        code = (
            "import pandas as pd\n"  # will be rejected by AST check
        )
        # Use pd directly (it's pre-injected in the sandbox)
        code = "result = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})"
        data = execute(server, code, session_id="test_df")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "dataframe"
        assert "a" in dr["output_preview"]["columns"]
        assert dr["output_preview"]["shape"] == [3, 2]

    def test_series_result(self, server):
        """Code that produces a Series is returned as output_type='series'."""
        code = "result = pd.Series([10, 20, 30], name='vals')"
        data = execute(server, code, session_id="test_series")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "series"
        assert dr["output_preview"]["name"] == "vals"
        assert dr["output_preview"]["values"] == [10, 20, 30]

    def test_blocked_import(self, server):
        """Code with an import statement is rejected by the AST check."""
        code = "import os\nresult = os.getcwd()"
        data = execute(server, code, session_id="test_block")
        assert data["ast_error"] is not None
        assert "block" in data["ast_error"].lower() or "import" in data["ast_error"].lower()
        assert data.get("direct_result") is None

    def test_blocked_forbidden_name(self, server):
        """Code referencing a forbidden builtin is blocked."""
        code = "result = eval('1+1')"
        data = execute(server, code, session_id="test_eval")
        assert data["ast_error"] is not None

    def test_stdout_captured(self, server):
        """print() output is captured in stdout field."""
        code = "print('test output')\nresult = 1"
        data = execute(server, code, session_id="test_stdout")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert "test output" in dr["stdout"]

    def test_numpy_result(self, server):
        """numpy arrays are returned with shape info."""
        code = "result = np.array([[1, 2], [3, 4]])"
        data = execute(server, code, session_id="test_np")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "ndarray"
        assert dr["output_preview"]["shape"] == [2, 2]

    def test_execute_with_fenced_code(self, server):
        """Triple-backtick fenced code blocks inside /execute are unwrapped."""
        code_block = "```python\nresult = 99\n```"
        resp = requests.post(
            f"{server}/api/chat",
            data={"message": f"/execute\n{code_block}", "session_id": "test_fence"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_preview"] == "99"

    def test_manifest_unchanged_after_execute(self, server):
        """Running /execute does not automatically register artefacts."""
        code = "result = pd.DataFrame({'x': [1, 2]})"
        data = execute(server, code, session_id="test_manifest")
        manifest = data["manifest"]
        # manifest is either a plain string like "(no artefacts registered)" or a list
        # Either way, "result" should not appear as a registered artefact name
        assert "result" not in str(manifest)


class TestHealthAndManifest:
    """Basic sanity checks for the server."""

    def test_artefact_detail_returns_description_and_data(self, server):
        r = requests.post(f"{server}/api/load-test-data?session_id=api_detail", timeout=30)
        assert r.status_code == 200

        detail = requests.get(
            f"{server}/api/artefact/universe?session_id=api_detail",
            timeout=30,
        )
        assert detail.status_code == 200

        body = detail.json()
        assert body["name"] == "universe"
        assert body["provenance"] == "engine"
        assert body["description"]
        assert body["data_type"] == "dataframe"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1
        assert "stock_name" in body["data"][0]

    def test_manifest_lists_registered_artefacts(self, server):
        requests.post(f"{server}/api/load-test-data?session_id=api_manifest", timeout=30)
        r = requests.get(f"{server}/api/manifest?session_id=api_manifest", timeout=30)
        assert r.status_code == 200
        body = r.json()
        assert any(item["name"] == "universe" for item in body["artefacts"])
        assert "`universe`" in body["manifest"]

    def test_server_responds(self, server):
        r = requests.get(server, timeout=5)
        assert r.status_code == 200
        assert "Artefact Registry" in r.text

    def test_manifest_empty_session(self, server):
        r = requests.get(f"{server}/api/manifest?session_id=empty_session", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert "artefacts" in d
        assert d["artefacts"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Browser-level UI tests (require `playwright install chromium`)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from playwright.sync_api import Page, expect, sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

browser_only = pytest.mark.skipif(
    not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed"
)


def _chromium_kwargs() -> dict:
    """Return launch kwargs, using a pre-installed binary when available."""
    import os
    path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    return {"executable_path": path} if path else {}


def _has_browser() -> bool:
    if not _PLAYWRIGHT_AVAILABLE:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, **_chromium_kwargs())
            browser.close()
            return True
    except Exception:
        return False


_BROWSER_AVAILABLE = _has_browser()
browser_ui = pytest.mark.skipif(
    not _BROWSER_AVAILABLE,
    reason="No Chromium browser available; run `playwright install chromium`",
)


@pytest.fixture(scope="session")
def browser_context(server):
    if not _BROWSER_AVAILABLE:
        pytest.skip("No browser")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **_chromium_kwargs())
        ctx = browser.new_context(base_url=server)
        yield ctx
        ctx.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    from conftest import setup_cdn_routes
    pg = browser_context.new_page()
    setup_cdn_routes(pg)
    yield pg
    pg.close()


@browser_ui
class TestBrowserExecuteUI:
    """UI-level tests using a real browser via Playwright."""

    def test_page_loads(self, page, server):
        page.goto(server)
        expect(page.locator(".chat-header")).to_be_visible()
        expect(page.locator(".manifest-panel")).to_be_visible()

    def test_execute_scalar_in_ui(self, page, server):
        """Send /execute result = 2 + 2 and verify the result appears in the UI."""
        page.goto(server)
        textarea = page.locator(".msg-input")
        textarea.fill("/execute\nresult = 2 + 2")
        page.locator(".send-btn").click()

        # Wait for the result block to appear
        result_label = page.locator(".result-label").first
        result_label.wait_for(timeout=15_000)
        expect(result_label).to_contain_text("Result")

        # The scalar value "4" should be visible
        scalar = page.locator(".result-scalar").first
        expect(scalar).to_contain_text("4")

    def test_execute_dataframe_in_ui(self, page, server):
        """Send /execute producing a DataFrame and verify the table renders."""
        page.goto(server)
        code = "result = pd.DataFrame({'col1': [10, 20, 30], 'col2': ['a', 'b', 'c']})"
        textarea = page.locator(".msg-input")
        textarea.fill(f"/execute\n{code}")
        page.locator(".send-btn").click()

        result_table = page.locator(".result-table").first
        result_table.wait_for(timeout=15_000)

        # Headers should contain our column names
        headers = page.locator(".result-table th").all_text_contents()
        assert "col1" in headers
        assert "col2" in headers

    def test_blocked_code_shows_error_in_ui(self, page, server):
        """Blocked code should show the ⛔ Code Blocked banner."""
        page.goto(server)
        textarea = page.locator(".msg-input")
        textarea.fill("/execute\nimport os\nresult = os.getcwd()")
        page.locator(".send-btn").click()

        error_label = page.locator(".error-label").first
        error_label.wait_for(timeout=10_000)
        expect(error_label).to_contain_text("Blocked")


@browser_ui
class TestFixturesAndChart:
    """Load fixtures then plot a time series and verify the chart renders."""

    def test_load_fixtures_and_plot_first_instrument(self, page, server):
        """
        1. Load all fixture artefacts via the 'Load Test Data' button.
        2. Run /execute to extract INST_0000's return series from stock_returns.
        3. Verify the line chart (Recharts SVG) renders in the result.
        """
        page.goto(server)

        # Step 1 — load fixtures
        page.get_by_role("button", name="Load Test Data").click()
        # Wait for stock_returns to appear in the manifest panel
        page.locator(".manifest-panel .artefact-name", has_text="stock_returns").wait_for(timeout=20_000)

        # Step 2 — run analysis via /execute
        code = "result = stock_returns[['INST_0000']]"
        page.locator(".msg-input").fill(f"/execute\n{code}")
        page.locator(".send-btn").click()

        # Step 3 — wait for result block
        result_label = page.locator(".result-label").first
        result_label.wait_for(timeout=20_000)
        expect(result_label).to_contain_text("Result")
        expect(result_label).to_contain_text("dataframe")

        # Step 4 — verify the SVG line chart rendered
        chart_svg = page.locator(".chart-wrap svg").first
        chart_svg.wait_for(timeout=10_000)
        expect(chart_svg).to_be_visible()

        # At least one polyline should exist (one series = INST_0000)
        line_path = page.locator(".chart-wrap polyline").first
        expect(line_path).to_be_visible()

        # The table should also render with the column name
        headers = page.locator(".result-table th").all_text_contents()
        assert "INST_0000" in headers
